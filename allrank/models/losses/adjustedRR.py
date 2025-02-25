import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.losses.loss_utils import deterministic_neural_sort, sinkhorn_scaling, stochastic_neural_sort
from allrank.models.metrics import dcg
from allrank.models.model_utils import get_torch_device
import math
import torch.nn.functional as F


def adjustedRR(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
               stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    MRR loss for Learning to Rank.
    Works best when there is exactly one relevant document per query.
    """

    y_pred = y_pred.clone()
    y_true = y_true.clone()
    
    # Mask out padding values
    mask = y_true != padded_value_indicator
    y_pred = y_pred * mask
    y_true = y_true * mask
    
    if log_scores:
        y_pred = torch.log1p(y_pred)  # log transformation for better numerical stability
    
    # Compute normal rank using softmax-based sorting relaxation
    pairwise_diffs = y_pred.unsqueeze(-1) - y_pred.unsqueeze(-2)
    normal_rank = 1 + torch.sum(F.sigmoid(pairwise_diffs / temperature), dim=-1)
    
    # Compute reciprocal rank
    rr = 1.0 / normal_rank
    
    # Bayesian adjustment
    weight_rr = normal_rank / (normal_rank + beta)
    weight_nr = beta / (normal_rank + beta)
    
    transformed_nr = torch.log1p(normal_rank)  # Log transformation to balance scale
    adjusted_rr = weight_rr * transformed_nr + weight_nr * (1 - rr)
    
    # Compute final loss (mean over non-padded items)
    loss = torch.mean(adjusted_rr[mask])
    return loss
