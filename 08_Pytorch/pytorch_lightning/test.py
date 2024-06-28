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
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        default_root_dir=cfg.callbacks.model_checkpoint.dirpath
    )

    data_module = DataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        transform=transforms.ToTensor()
    )
    model = CNN(
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        dropout_ratio=cfg.model.dropout_ratio,
        use_scheduler=cfg.model.use_scheduler
    )

    trainer.test(
        model,
        data_module,
        ckpt_path=cfg.test.weight_file,
        verbose=True
    )

if __name__ == "__main__":
    main()