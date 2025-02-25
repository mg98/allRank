from urllib.parse import urlparse
import sys

import allrank.models.losses as losses
import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
import wandb

def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--user-id", help="User id")
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")
    parser.add_argument("--wandb", help="If true, log to wandb", action=BooleanOptionalAction)

    return parser.parse_args()


def run():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    if args.user_id: config.data.path = f"/var/scratch/mgregori/datasets/tribler_crawl/by_user/{args.user_id}/_normalized"
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size)

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Calculate the size of the model in MB
    param_size_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    param_size_mb = param_size_bytes / (1024 ** 2)
    print(f"Model size: {param_size_mb:.2f} MB")
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Train dataset shape: {train_ds.shape}")
    print(f"Validation dataset shape: {val_ds.shape}")

    if args.wandb:
        wandb.init(project=config.wandb_project_id, config=asdict(config), name=args.user_id if args.user_id else None)
        for metric, ks in config.metrics.items():
            for k in ks:
                wandb.define_metric(f"{metric}_{k}", summary="max")

    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training
        result = fit(
            model=model,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            wandb_logging=args.wandb,
            **asdict(config.training)
        )

    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)
    
    if args.wandb:
        abs_output_dir = os.path.abspath(paths.output_dir)

        artifact = wandb.Artifact(name="experiment-results", type="results", description="Experiment results")
        artifact.add_reference(f"file://{abs_output_dir}")
        wandb.log_artifact(artifact)

        artifact = wandb.Artifact(name="final-model", type="model", description="Final model")
        artifact.add_reference(f"file://{abs_output_dir}/model.pkl")
        wandb.log_artifact(artifact)
        
        wandb.finish()

    assert_expected_metrics(result, config.expected_metrics)


if __name__ == "__main__":
    run()
