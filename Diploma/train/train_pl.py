import argparse
from losses import CCCLoss, CrossEntropyLoss
from time import strftime
from datasets.affwild2_dataset import AffWildVADataset
from datasets.affwild2_audio_dataset import AffWildVADatasetWithAudio
from datasets.affwild2_rnn_dataset import AffWildVADatasetRNN
import yaml

from munch import munchify
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from metrics import CCC_score
from models.resnet50 import create_model as create_resnet50_model
from models.fer_plus_resnet50 import create_model as create_resnet50_fer_model
from transforms import train_transforms, val_transforms
from torchsummary import summary
from pretrained_models.resnet50_ferplus_dag_audio import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--cfg', '-c', required=True, help='Path to .yaml config file')
    return parser.parse_args()


class EmotionModel(pl.LightningModule):
    def __init__(self, cfg):
        super(EmotionModel, self).__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.train_transform = train_transforms(self.cfg)
        self.val_transform = val_transforms(self.cfg)
        if self.cfg.model_name == "resnet50":
            self.model = create_resnet50_model(cfg)
        elif self.cfg.model_name == "resnet50_ferplus_dag":
            self.model = create_resnet50_fer_model(cfg)
        else:
            raise NotImplementedError
        print(self.model)
        if not self.cfg.rnn:
            input_size = [(3, self.cfg.height, self.cfg.width)]
            if "audio_path" in self.cfg:
                input_size.append((1, 122, 122))
        else:
            input_size = [(self.cfg.seq_len, 3, self.cfg.height, self.cfg.width)]
            if "audio_path" in self.cfg:
                input_size.append((self.cfg.seq_len, 1, 122, 122))
        summary(self.model, input_size=input_size, device="cpu")
        self.criterion_ccc = CCCLoss(self.cfg.digitize_number)
        self.criterion_ce = CrossEntropyLoss(self.cfg.digitize_number)
        if "ce_weight" in self.cfg:
            self.ce_weight = self.cfg.ce_weight
        else:
            self.ce_weight = 0.5
        self.bins = torch.linspace(-1, 1, steps=self.cfg.digitize_number).to(self.device)

    def on_fit_start(self):
        pl.seed_everything(42)
    # def class_pred_to_number(self, predictions):
    #     return torch.true_divide(predictions - self.cfg.num_levels, self.cfg.num_levels)
    #
    # def number_to_class(self, number):
    #     return torch.round(number * self.cfg.num_levels + self.cfg.num_levels).long()

    def train_dataloader(self):
        if self.cfg.rnn:
            if "audio_path" in self.cfg:
                raise NotImplementedError("Wait for it")
            else:
                dataset = AffWildVADatasetRNN(self.cfg.data_path, self.cfg.labels, self.train_transform, self.cfg.seq_len)
        else:
            if "audio_path" in self.cfg:
                dataset = AffWildVADatasetWithAudio(self.cfg.data_path, self.cfg.labels, self.cfg.audio_path,
                                                             self.train_transform)
            else:
                dataset = AffWildVADataset(self.cfg.data_path, self.cfg.labels,
                                           self.train_transform)
        return DataLoader(dataset, batch_size=self.cfg.train_batch_size, num_workers=self.cfg.train_workers,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        if self.cfg.rnn:
            if "audio_path" in self.cfg:
                raise NotImplementedError("Wait for it")
            else:
                dataset = AffWildVADatasetRNN(self.cfg.data_path, self.cfg.labels, self.train_transform,
                                              self.cfg.seq_len, mode='val')
        else:
            if "audio_path" in self.cfg:
                dataset = AffWildVADatasetWithAudio(self.cfg.data_path, self.cfg.labels, self.cfg.audio_path,
                                                    self.train_transform, mode="val")
            else:
                dataset = AffWildVADataset(self.cfg.data_path, self.cfg.labels,
                                           self.train_transform, mode='val')
        return DataLoader(dataset, batch_size=self.cfg.test_batch_size, num_workers=self.cfg.test_workers,
                          shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

    def flatten_rnn_pred(self, val_pred, arousal_pred, val_target, arousal_target):
        B, S, C = val_pred.size()
        val_target = val_target.view(B * S, -1)
        arousal_target = arousal_target.view(B * S, -1)
        val_pred = val_pred.view(B * S, C)
        arousal_pred = arousal_pred.view(B * S, C)
        return val_pred, arousal_pred, val_target, arousal_target


    def calculate_loss(self, outputs, targets):
        val_target = targets['valence']
        arousal_target = targets['arousal']
        val_pred = outputs['val_pred']
        arousal_pred = outputs['arousal_pred']
        # valence_class = self.number_to_class(valence)
        # arousal_class = self.number_to_class(arousal)
        if self.cfg.rnn:
            val_pred, arousal_pred, val_target, arousal_target = self.flatten_rnn_pred(val_pred, arousal_pred, val_target, arousal_target)
        loss_ccc_v = self.criterion_ccc(val_pred, val_target)
        loss_ccc_a = self.criterion_ccc(arousal_pred, arousal_target)
        loss_ce_v = self.criterion_ce(val_pred, val_target)
        loss_ce_a = self.criterion_ce(arousal_pred, arousal_target)
        loss_ce = self.ce_weight * (loss_ce_a + loss_ce_v)
        loss_ccc = (1 - self.ce_weight) * (loss_ccc_a + loss_ccc_v)
        loss = loss_ce + loss_ccc
        return loss_ce, loss_ccc, loss

    def calculate_metrics(self, outputs, targets):
        val_target = targets['valence']
        arousal_target = targets['arousal']
        val_pred = outputs['val_pred']
        arousal_pred = outputs['arousal_pred']
        if self.cfg.rnn:
            val_pred, arousal_pred, val_target, arousal_target = self.flatten_rnn_pred(val_pred, arousal_pred,
                                                                                       val_target, arousal_target)
        val_pred = val_pred.softmax(-1)
        if not self.bins.device == val_pred.device:
            self.bins = self.bins.to(val_pred.device)
        val_pred = (self.bins * val_pred).sum(-1)
        valence_CCC = CCC_score(val_pred, val_target)
        arousal_pred = arousal_pred.softmax(-1)
        arousal_pred = (self.bins * arousal_pred).sum(-1)
        arousal_CCC = CCC_score(arousal_pred, arousal_target)
        return valence_CCC, arousal_CCC

    def training_step(self, batch, batch_idx):
        images = batch['image'].float()
        if "audio" in batch:
            outputs = self.model(images, batch["audio"])
        else:
            outputs = self.model(images)
        loss_ce, loss_ccc, loss = self.calculate_loss(outputs, batch["targets"])
        valence_CCC, arousal_CCC = self.calculate_metrics(outputs, batch["targets"])
        total_CCC = valence_CCC + arousal_CCC
        # self.log('train/valence_CCC', valence_CCC, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        # self.log('train/arousal_CCC', arousal_CCC, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        self.log('train/total_ccc', total_CCC, prog_bar=True, logger=True, sync_dist=True)
        self.log('train/loss_ce', loss_ce, sync_dist=True)
        self.log('train/loss_ccc', loss_ccc, sync_dist=True)
        self.log('train/loss', loss, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images = batch['image'].float()
        if "audio" in batch:
            outputs = self.model(images, batch["audio"])
        else:
            outputs = self.model(images)
        loss_ce, loss_ccc, loss = self.calculate_loss(outputs, batch["targets"])
        valence_CCC, arousal_CCC = self.calculate_metrics(outputs, batch["targets"])
        total_CCC = valence_CCC + arousal_CCC

        self.log('val/valence_CCC', valence_CCC, prog_bar=True, sync_dist=True)
        self.log('val/arousal_CCC', arousal_CCC, prog_bar=True, sync_dist=True)
        self.log('val/total_ccc', total_CCC, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/loss', loss, prog_bar=True, logger=True, sync_dist=True)
        # self.log('val/loss_ce', loss_ce, prog_bar=True, logger=True, sync_dist=True)
        # self.log('val/loss_ccc', loss_ccc, prog_bar=True, logger=True, sync_dist=True)
        self.log('total_ccc', total_CCC, sync_dist=True)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        if hasattr(self.cfg, "milestones"):
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, self.cfg.milestones, self.cfg.gamma)
            return [opt], [sch]
        return [opt]





def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f)
    cfg = munchify(cfg)
    model = EmotionModel(cfg)
    if isinstance(cfg.gpus, int):
        gpus = [cfg.gpus]
    else:
        gpus = [int(gpu_id) for gpu_id in cfg.gpus.split(',')]
    base_log_dir = "logs"
    experiment_version = strftime("%d-%m-%Y-%H:%M:%S")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/total_ccc',
        filename='{epoch:02d}-{total_ccc:.3f}',
        mode='max',
        save_top_k=3
    )
    if 'checkpoint' in cfg:
        load_model_weights(model, cfg.checkpoint)


    tb_logger = pl.loggers.TensorBoardLogger(base_log_dir, name=cfg.experiment, version=experiment_version)
    trainer = pl.Trainer(gpus=gpus, max_epochs=cfg.epochs, logger=tb_logger, accelerator=cfg.accelerator, callbacks=[checkpoint_callback])
    if cfg.test:
        trainer.test(model)
        return
    trainer.fit(model)


if __name__ == '__main__':
    main()
