import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from argparse import ArgumentParser

# self-defined func
sys.path.append("..")
from utils.tool_simple import txt_to_list
from model.func import get_scheduler_batch

import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, Recall
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
    StochasticWeightAveraging,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# env
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch
# torch.multiprocessing.set_start_method('spawn')
# multiprocessing.set_start_method('spawn')

class Dataset_Sentiment(Dataset):
    def __init__(self, list_data, list_label, tokenizer, max_length_token=256):
        self.data = list_data
        self.label = list_label
        self.tokenizer = tokenizer
        self.max_length_token = max_length_token
        # assert len(self.data) == len(self.label)

    def __getitem__(self, index):
        tweet = self.data[index]
        label = int(self.label[index])
        # return tweet, label
        tweet = self.tokenizer(
            tweet,
            padding="max_length",
            truncation=True,
            max_length=self.max_length_token,
            return_tensors="pt",
        )
        return (
            tweet["input_ids"].squeeze(),
            tweet["attention_mask"].squeeze(),
            torch.LongTensor([label]),
        )

    def __len__(self):
        return len(self.data)


def loading_tweet_label(path_dir_dataset, dataset_type, tokenizer):
    list_tweet = txt_to_list(os.path.join(path_dir_dataset, f"{dataset_type}_text.txt"))
    list_label = txt_to_list(
        os.path.join(path_dir_dataset, f"{dataset_type}_labels.txt")
    )
    return list_tweet, list_label


class SemEval_Sentiment(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        data_dir: str,
        batch_size_train: int = 32,
        batch_size_eval: int = 256,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.tokenizer = tokenizer

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        list_tweet, list_label = loading_tweet_label(
            self.data_dir, "train", self.tokenizer
        )
        print(f"Loading: {len(list_tweet)} training cases")
        self.dataset_train = Dataset_Sentiment(
            list_tweet, list_label, tokenizer=self.tokenizer, max_length_token=256
        )

        list_tweet, list_label = loading_tweet_label(
            self.data_dir, "val", self.tokenizer
        )
        print(f"Loading: {len(list_tweet)} eval cases")
        self.dataset_val = Dataset_Sentiment(
            list_tweet, list_label, tokenizer=self.tokenizer, max_length_token=256
        )

        list_tweet, list_label = loading_tweet_label(
            self.data_dir, "test", self.tokenizer
        )
        print(f"Loading: {len(list_tweet)} testing cases")
        self.dataset_test = Dataset_Sentiment(
            list_tweet, list_label, tokenizer=self.tokenizer, max_length_token=256
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size_train,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )



class Sentiment(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.path_model = self.args.path_model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path_model, num_labels=args.num_label, return_dict=True
        )
        self.loss = nn.CrossEntropyLoss()
        self.metric_acc = Accuracy()
        self.metric_f1 = F1Score(num_classes=args.num_label, average="macro")
        self.metric_recall = Recall(average="macro", num_classes=args.num_label)

    # model out
    def forward(self, batch_input, batch_mask):
        out = self.model(input_ids=batch_input, attention_mask=batch_mask)
        pred = out["logits"]
        return pred

    # step template
    def step(self, batch, batch_idx):
        batch_input, batch_mask, batch_label = batch
        batch_pred = self(batch_input, batch_mask)
        batch_label = batch_label.squeeze()
        loss = self.loss(batch_pred, batch_label)
        return {
            "loss": loss,
            "batch_pred": batch_pred.detach(),
            "batch_label": batch_label.detach(),
        }
    # step_end template
    def step_end(self, outputs, label):
        step_acc = self.metric_acc(outputs["batch_pred"], outputs["batch_label"]).item()
        step_f1 = self.metric_f1(outputs["batch_pred"], outputs["batch_label"]).item()
        step_recall = self.metric_recall(outputs["batch_pred"], outputs["batch_label"]).item()
        self.log(f"{label}_acc", step_acc, prog_bar=True, sync_dist=True)
        self.log(f"{label}_f1", step_f1, prog_bar=True, sync_dist=True)
        self.log(f"{label}_recall", step_recall, prog_bar=True, sync_dist=True)
        return {
            "loss": outputs["loss"].mean(),
            f"{label}_acc": step_acc,
            f"{label}_f1": step_f1,
            f"{label}_recall": step_f1,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_step_end(self, outputs):
        return self.step_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step_end(self, outputs):
        return self.step_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.step_end(outputs, "test")

    # 定义optimizer,以及可选的lr_scheduler
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        # filter(lambda p: p.requires_grad, model_clf.parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        # torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        # No scheduler
        if self.args.scheduler == None:
            print(f"No scheduler and fixed lr={str(self.args.learning_rate)}")
            return optimizer
        else:
            scheduler = get_scheduler_batch(optimizer, self.args.scheduler, lr_base=self.args.learning_rate, num_epoch=self.args.epoch_max, num_batch=self.args.num_batch_train)
            print(
                f"scheduler:{self.args.scheduler} and lr={str(self.args.learning_rate)}"
            )
            return ([optimizer], [{"scheduler": scheduler, "interval": "step"}])

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        return parser


class argparse:
    pass


def main(args, name_experiment):
    pl.seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.path_model)
    data_sentiment = SemEval_Sentiment(
        tokenizer=tokenizer,
        data_dir=args.path_dataset,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        num_workers=4,
    )
    data_sentiment.setup()
    args.num_batch_train = len(data_sentiment.train_dataloader()) // len(args.devices) + 1
    print(f"Training: num_epoch:{args.epoch_max}, num_batch:{args.num_batch_train}, all:{args.epoch_max*args.num_batch_train}")

    model_sentiment = Sentiment(args=args)

    checkpoint = ModelCheckpoint(
        filename="{epoch:02d}-{val_f1:.4f}" if "emotion" in args.path_dataset else "{epoch:02d}-{val_recall:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        monitor=args.metric,
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor=args.metric, patience=10, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # device_stats = DeviceStatsMonitor()
    callbacks = [checkpoint, early_stopping, lr_monitor]
    if args.use_swa == True:
        callbacks.append(StochasticWeightAveraging())

    logger = TensorBoardLogger("lightning_logs", name=name_experiment)

    print("hparams.auto_lr_find=", args.auto_lr_find)
    if args.auto_lr_find:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],
            # callbacks=callbacks,
            max_epochs=args.epoch_max,
            min_epochs=args.epoch_min,
            val_check_interval=args.val_check_interval,
            gradient_clip_val=args.gradient_clip_val,
            deterministic=True,
            logger=logger,
            log_every_n_steps=10,
            # profiler="simple",
        )
        # 搜索学习率范围
        lr_finder = trainer.tuner.lr_find(
            model_sentiment,
            datamodule=data_sentiment,
        )
        fig = lr_finder.plot(suggest=True)
        plt.savefig("lr_find.svg", dpi=300)
        lr = lr_finder.suggestion()
        print("suggest lr=", lr)
        model_sentiment.hparams.learning_rate = lr
        args.learning_rate = lr
        del trainer
        del model_sentiment
        model_sentiment = Sentiment(args, learning_rate=args.learning_rate)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_false",
        callbacks=callbacks,
        max_epochs=args.epoch_max,
        min_epochs=args.epoch_min,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=True,
        logger=logger,
        log_every_n_steps=1,
        # profiler="simple",
    )


    # 训练模型
    trainer.fit(model_sentiment, data_sentiment)

    # the best
    print(f"best_model_path: ", trainer.checkpoint_callback.best_model_path)
    print(f"best_model_score: ", trainer.checkpoint_callback.best_model_score)
    # testing
    # test_result = pl.Trainer(accelerator="gpu",devices=args.devices[0]).test(model_sentiment, data_sentiment.test_dataloader())
    test_result = trainer.test(model_sentiment, data_sentiment.test_dataloader(), ckpt_path='best', verbose=True)
    print(test_result)
    # save
    model_sentiment.model.save_pretrained(args.path_out)
    tokenizer.save_pretrained(args.path_out)


if __name__ == "__main__":
    args = argparse()
    args.seed = 42
    args.path_model = "bert-base-uncased"
    args.path_dataset = ""
    args.path_out = ""
    args.batch_size_train = 32
    args.batch_size_eval = 500
    args.val_check_interval = 0.2
    args.epoch_max = 10
    args.epoch_min = 1
    args.accumulation = 1
    args.gradient_clip_val = 1.0
    args.learning_rate = 5e-5
    args.weight_decay = 0.01
    args.adam_epsilon = 1e-8
    args.auto_lr_find = False
    # suggest lr= 2.2908676527677735e-07
    args.scheduler = 'CyclicLR'
    # StepLR, MultiStepLR, LambdaLR, ExponentialLR, OneCycleLR, CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    args.use_swa = False
    args.num_label = 4 if "emotion" in args.path_dataset else 3
    args.metric = 'val_f1' if "emotion" in args.path_dataset else 'val_recall'

    # devices
    # args.devices = [0,1,2,3,4,5,6,7]
    args.devices = [1,2,3]
    # args.devices = [4,5,6,7]

    # name_experiment
    name_experiment = ''
    # main 
    main(args, name_experiment)