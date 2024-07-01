## python train.py -m "criterion._target_=torch.nn.CrossEntropyLoss" "criterion._target_=torch.nn.MSELoss"

import os
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra

from omegaconf import DictConfig
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from models.model import CNN
from data.dataset import DataModule

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    csv_logger = CSVLogger(save_dir=cfg.logger.csv.save_dir, name=cfg.logger.csv.name)
    tb_logger = TensorBoardLogger(save_dir=cfg.logger.tensorboard.save_dir, name=cfg.logger.tensorboard.name)

    early_stop_callback = EarlyStopping(
        monitor=cfg.callbacks.early_stop.monitor,
        mode=cfg.callbacks.early_stop.mode
    )
    save_ckpt_callback = ModelCheckpoint(
        dirpath=cfg.callbacks.model_checkpoint.dirpath,
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        filename=cfg.callbacks.model_checkpoint.filename,
        save_last=cfg.callbacks.model_checkpoint.save_last,
        save_weights_only=cfg.callbacks.model_checkpoint.save_weights_only,
        verbose=cfg.callbacks.model_checkpoint.verbose,
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k
    )

    data_module = DataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        transform=transforms.ToTensor()
    )
    model = CNN(cfg)

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        callbacks=[early_stop_callback, save_ckpt_callback],
        logger=[csv_logger, tb_logger],
        default_root_dir=cfg.callbacks.model_checkpoint.dirpath
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
