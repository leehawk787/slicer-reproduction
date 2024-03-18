
from argparse import ArgumentParser, Namespace

from lightning import seed_everything
import torch
from data.slicer_lightning_data_module import SlicerLightningDataModule
from modules.slicer_lightning_module import SlicerLightningModule
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def read_hyperparameters() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--dataset", default="conll")
    parser.add_argument("--deterministic", default=True)
    parser.add_argument("--fast_dev_run", default=False)
    parser.add_argument("--slice_size", default=4)
    #parser.add_argument("--num_workers", default=4)
    parser.add_argument("--offline", default=True)
    parser.add_argument("--random_seed", default=42)
    parser.add_argument("--slicer", default=True)
    parser.add_argument("--save_dir", default="")
    args = parser.parse_args()
    return args

def create_experiment_name(args):
    base_name = 'Slicer' if args.slicer else 'Vanilla'
    dataset = str(args.dataset)
    batch_size = str(args.batch_size)
    num_slices = str(args.slice_size)
    return 'NER_' + base_name \
        + "-Dataset:" + dataset \
        + "-BatchSize:" + batch_size \
        + "-Slices:" + num_slices \

def main(hparams):
    # Random seeding for reproducablity
    seed_everything(seed=hparams.random_seed, workers=True)

    # float precision
    torch.set_float32_matmul_precision('high')

    # Init logging
    experiment_name = create_experiment_name(hparams)

    wandb_logger = WandbLogger(
        log_model=False,
        project="MLNLP_Slicer", 
        name=experiment_name,
        save_dir= hparams.save_dir,
        checkpoint_name= experiment_name,
        offline=hparams.offline
        )

    datamodule = SlicerLightningDataModule(        
        batch_size=hparams.batch_size, 
        num_workers = hparams.num_workers)
    
    # important to call here, otherwise, the name of the test datasets will not be set causing erors in the module
    datamodule.setup(stage="train_"+hparams.dataset)

    # Init the building blocks
    module = SlicerLightningModule(
        test_datasets_names=datamodule.test_datasets_names,
        n_labels=7,
        learning_rate=2e-5,
        weight_decay=0.05,
        is_slicer=hparams.slicer,
        slice_size= hparams.slice_size
    )

    # Trainer (https://lightning.ai/docs/pytorch/stable/common/trainer.html)
    trainer = pl.Trainer(accelerator=hparams.accelerator,
                         max_epochs=10 if hparams.dataset == "conll" else 5,
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         fast_dev_run=hparams.fast_dev_run,
                         deterministic=hparams.deterministic,
                         )

    # Fit the model (and evaluate on validation data as defined)
    trainer.fit(module, datamodule=datamodule)
    # trainer.save_checkpoint("./../checkpoints/" + experiment_name + "/example.ckpt")

    # Test model
    if not hparams.fast_dev_run:
        trainer.test(datamodule=datamodule, ckpt_path='last')

    del module
    del datamodule
    del trainer

    torch.cuda.empty_cache()
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    args = read_hyperparameters()
    main(args)
