from typing import Tuple, Dict, List, Generator

import torch
from torch.utils.data.dataloader import DataLoader

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import LibSVMDataset
from allrank.models.metrics import ndcg, dcg
from allrank.models.model import LTRModel
from allrank.models.model_utils import get_torch_device


def rank_slates(datasets: Dict[str, LibSVMDataset], model: LTRModel, config: Config) \
        -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Ranks given datasets according to a given model

    :param datasets: dictionary of role -> dataset that will be ranked
    :param model: a model to use for scoring documents
    :param config: config for DataLoaders
    :return: dictionary of role -> ranked dataset
        every dataset is a Tuple of torch.Tensor - storing X and y in the descending order of the scores.
    """

    dataloaders = {role: __create_data_loader(ds, config) for role, ds in datasets.items()}

    ranked_slates = {role: __rank_slates(dl, model) for role, dl in dataloaders.items()}

    return ranked_slates


def __create_data_loader(ds: LibSVMDataset, config: Config) -> DataLoader:
    return DataLoader(ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)


def __rank_slates(dataloader: DataLoader, model: LTRModel) -> Tuple[torch.Tensor, torch.Tensor]:
    reranked_X = []
    reranked_y = []
    model.eval()
    device = get_torch_device()
    with torch.no_grad():
        for xb, yb, indices, qids in dataloader:
            qids = qids.squeeze(-1)

            X = xb.type(torch.float32).to(device=device)
            y_true = yb.to(device=device)
            indices = indices.to(device=device)

            input_indices = torch.ones_like(y_true).type(torch.long)
            mask = (y_true == losses.PADDED_Y_VALUE)
            scores = model.score(X, mask, input_indices)

            # Iterate over each query in the batch
            for i in range(scores.size(0)):
                query_id = qids[i].item()
                slate_scores = scores[i]
                slate_labels = y_true[i]
                slate_indices = indices[i]
                slate_mask = mask[i]
                
                valid_scores = slate_scores[~slate_mask]
                valid_labels = slate_labels[~slate_mask]
                valid_indices = slate_indices[~slate_mask]
                
                if valid_scores.numel() == 0:
                    print(f"Query ID: {query_id} has no valid documents.")
                    continue
                
                # Optionally, compute the rankings
                sorted_scores, sorted_idx = torch.sort(valid_scores, descending=True)
                sorted_labels = valid_labels[sorted_idx]
                sorted_original_indices = valid_indices[sorted_idx]
                
                # Debugging output
                print(f"Query ID: {query_id}")
                print(f"{'Rank':<5}{'Score':<10}{'Label':<10}{'Original Index':<15}")
                for rank, (score, label, idx) in enumerate(zip(sorted_scores, sorted_labels, sorted_original_indices), start=1):
                    print(f"{rank:<5}{score.item():<10.4f}{int(label.item()):<10}{int(idx.item()):<15}")
                print("-" * 40)

            scores[mask] = float('-inf')
            
            _, indices = scores.sort(descending=True, dim=-1)
            indices_X = torch.unsqueeze(indices, -1).repeat_interleave(X.shape[-1], -1)
            reranked_X.append(torch.gather(X, dim=1, index=indices_X).cpu())
            reranked_y.append(torch.gather(y_true, dim=1, index=indices).cpu())

    combined_X = torch.cat(reranked_X)
    combined_y = torch.cat(reranked_y)
    return combined_X, combined_y


def __clicked_ndcg(ordered_clicks: List[int]) -> float:
    return ndcg(torch.arange(start=len(ordered_clicks), end=0, step=-1, dtype=torch.float32)[None, :],
                torch.tensor(ordered_clicks)[None, :]).item()


def __clicked_dcg(ordered_clicks: List[int]) -> float:
    return dcg(torch.arange(start=len(ordered_clicks), end=0, step=-1, dtype=torch.float32)[None, :],
               torch.tensor(ordered_clicks)[None, :]).item()


def metrics_on_clicked_slates(clicked_slates: Tuple[List[torch.Tensor], List[List[int]]]) \
        -> Generator[Dict[str, float], None, None]:
    Xs, ys = clicked_slates
    for X, y in zip(Xs, ys):
        yield {
            "slate_length": len(y),
            "no_of_clicks": sum(y > 0),  # type: ignore
            "dcg": __clicked_dcg(y),
            "ndcg": __clicked_ndcg(y)
        }
