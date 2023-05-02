import os
import sys
import numpy as np
from scipy.special import softmax
from sklearn.metrics import *
# torch
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryAUROC, BinaryAveragePrecision
# pytorch
import pytorch_lightning as pl

# transformers
from transformers import *
from transformers import AdamW, get_linear_schedule_with_warmup
# some defined method
sys.path.append("..")
from model.func import get_scheduler_batch

class trainer_DCT(pl.LightningModule):
    def __init__(self, args, model, data):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'data'])
        self.args = args
        self.model = model
        self.data = data
        # loss
        self.criterion_loss_clf = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_loss_distill = nn.MSELoss(reduction="mean")
        # metric
        self.metric_auroc = BinaryAUROC()
        self.metric_ap = BinaryAveragePrecision()
        self.metric_acc = BinaryAccuracy()
        self.metric_recall = BinaryRecall()
        self.metric_f1 = BinaryF1Score()
        # self.metric_acc = Accuracy()
        # self.metric_f1 = F1Score(num_classes=2, average="macro")
        # self.metric_recall = Recall(num_classes=2, average="macro")

    # model out
    def forward(self, batch_input, batch_mask, batch_len):
        out = self.model(batch_input, batch_mask, batch_len)
        logits = out["logits"]
        emb_teacher = out["emb_teacher"]
        emb_student = out["emb_student"]
        scores_attn = out["scores_attn"]
        return logits, emb_teacher, emb_student, scores_attn

    # step template
    def step(self, batch, batch_idx):
        batch_input, batch_mask, batch_label, batch_len = batch
        batch_logits, batch_emb_teacher, batch_emb_student, scores_attn = self(batch_input, batch_mask, batch_len)
        if batch_logits.dim() == 1:
            batch_logits = batch_logits.unsqueeze(0)
        if batch_label.dim() > 1:
            batch_label = batch_label.squeeze()
        # loss
        loss_clf = self.criterion_loss_clf(batch_logits, batch_label)
        loss_distill = self.criterion_loss_distill(batch_emb_teacher, batch_emb_student)
        loss = loss_clf + (loss_distill * self.args.weight_clf)
        return {
            "loss": loss,
            "loss_clf": loss_clf,
            "loss_distill": loss_distill,
            "batch_logits": batch_logits.detach(),
            "batch_label": batch_label.detach(),
            "scores_attn": scores_attn.detach(),
        }
    # metrics template
    def step_metric(self, pred, target, label):
        step_auroc = self.metric_auroc(pred, target).item()
        step_ap = self.metric_ap(pred, target).item()
        step_acc = self.metric_acc(pred, target).item()
        step_f1 = self.metric_f1(pred, target).item()
        step_recall = self.metric_recall(pred, target).item()
        self.log(f"{label}_auroc", step_auroc, prog_bar=True, sync_dist=True)
        self.log(f"{label}_ap", step_ap, prog_bar=True, sync_dist=True)
        self.log(f"{label}_acc", step_acc, prog_bar=True, sync_dist=True)
        self.log(f"{label}_f1", step_f1, prog_bar=True, sync_dist=True)
        self.log(f"{label}_recall", step_recall, prog_bar=True, sync_dist=True)
    # step_end template
    def step_end(self, outputs, label):
        # metrics
        if label!='train':
            self.step_metric(outputs["batch_logits"][:, 1], outputs["batch_label"], label)
        # loss
        if label!='test':
            self.log(f"{label}_loss", outputs['loss'].detach().item(), prog_bar=False, sync_dist=True)
            self.log(f"{label}_loss_clf", outputs['loss_clf'].detach().item(), prog_bar=False, sync_dist=True)
            self.log(f"{label}_loss_distill", outputs['loss_distill'].detach().item(), prog_bar=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_step_end(self, outputs):
        self.step_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step_end(self, outputs):
        self.step_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        result = self.step(batch, batch_idx)
        logits = result["batch_logits"][:, 1]
        scores_attn = np.array(result['scores_attn'].cpu())
        path_file_scores = os.path.join('out_score', f'scores_attn_{batch_idx}.txt')
        # path_file_scores = f'scores_attn_{batch_idx}.txt'
        np.savetxt(path_file_scores, scores_attn, fmt='%.4f')
        print(f"Save scores of attn to {path_file_scores}")
        return result

    def test_step_end(self, outputs):
        self.step_end(outputs, "test")
    
    def predict_step(self, batch, batch_idx):
        result = self.test_step(batch, batch_idx)
        scores_attn = np.array(result['scores_attn'].cpu())
        path_file_scores = os.path.join('out_score', f'scores_attn_{batch_idx}.txt')
        # path_file_scores = f'scores_attn_{batch_idx}.txt'
        np.savetxt(path_file_scores, scores_attn, fmt='%.4f')
        print(f"Save scores of attn to {path_file_scores}")
        return result

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
        )
        # torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        # No scheduler
        if self.args.scheduler == None:
            print(f"No scheduler and fixed lr={str(self.args.learning_rate)}")
            return optimizer
        else:
            scheduler = get_scheduler_batch(optimizer, self.args.scheduler, lr_base=self.args.learning_rate, num_epoch=self.args.epochs/2, num_batch=self.args.num_batch_train)
            print(
                f"scheduler:{self.args.scheduler} and lr={str(self.args.learning_rate)}"
            )
            return ([optimizer], [{"scheduler": scheduler, "interval": "step"}])

    @staticmethod
    def add_model_args(parent_parser):
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--learning_rate", type=float, default=2e-5)
        # return parser
        pass

class trainer_DCT_self(trainer_DCT):
    def __init__(self, args, model, data):
        super().__init__(args, model, data)

    # model out
    def forward(self, batch_input, batch_mask, batch_len):
        out = self.model(batch_input, batch_mask, batch_len)
        logits = out["logits"]
        scores_attn = out["scores_attn"]
        return logits, scores_attn

    # step template
    def step(self, batch, batch_idx):
        batch_input, batch_mask, batch_label, batch_len = batch
        batch_logits, scores_attn = self(batch_input, batch_mask, batch_len)
        if batch_logits.dim() == 1:
            batch_logits = batch_logits.unsqueeze(0)
        if batch_label.dim() > 1:
            batch_label = batch_label.squeeze()
        loss = self.criterion_loss_clf(batch_logits, batch_label)
        return {
            "loss": loss,
            "batch_logits": batch_logits.detach(),
            "batch_label": batch_label.detach(),
            "scores_attn": scores_attn.detach(),
        }
    # step_end template
    def step_end(self, outputs, label):
        # metrics
        if label!='train':
            self.step_metric(outputs["batch_logits"][:, 1], outputs["batch_label"], label)
        # loss
        if label!='test':
            self.log(f"{label}_loss", outputs['loss'].detach().item(), prog_bar=False, sync_dist=True)

class trainer_DCT_emb_tweet(trainer_DCT):
    def __init__(self, args, model, data):
        super().__init__(args, model, data)

        # model out
    def forward(self, batch_emb_teacher, batch_input, batch_mask, batch_len):
        out = self.model(batch_emb_teacher, batch_input, batch_mask, batch_len)
        logits = out["logits"]
        emb_teacher = out["emb_teacher"]
        emb_student = out["emb_student"]
        scores_attn = out["scores_attn"]
        return logits, emb_teacher, emb_student, scores_attn

    # step template
    def step(self, batch, batch_idx):
        batch_emb_teacher, batch_input, batch_mask, batch_label, batch_len = batch
        batch_logits, batch_emb_teacher, batch_emb_student, scores_attn = self(batch_emb_teacher, batch_input, batch_mask, batch_len)
        if batch_logits.dim() == 1:
            batch_logits = batch_logits.unsqueeze(0)
        if batch_label.dim() > 1:
            batch_label = batch_label.squeeze()
        loss_clf = self.criterion_loss_clf(batch_logits, batch_label)
        loss_distill = self.criterion_loss_distill(batch_emb_teacher, batch_emb_student)
        loss = loss_clf + (loss_distill * self.args.weight_clf)
        return {
            "loss": loss,
            "loss_clf": loss_clf,
            "loss_distill": loss_distill,
            "batch_logits": batch_logits.detach(),
            "batch_label": batch_label.detach(),
            "scores_attn": scores_attn.detach(),
        }
    
class trainer_DCT_emb_merge(trainer_DCT):
    def __init__(self, args, model, data):
        super().__init__(args, model, data)

        # model out
    def forward(self, batch_emb_teacher, batch_input, batch_mask, batch_len):
        out = self.model(batch_emb_teacher, batch_input, batch_mask, batch_len)
        logits = out["logits"]
        scores_attn = out["scores_attn"]
        return logits, scores_attn

    # step template
    def step(self, batch, batch_idx):
        batch_emb_teacher, batch_input, batch_mask, batch_label, batch_len = batch
        batch_logits, scores_attn = self(batch_emb_teacher, batch_input, batch_mask, batch_len)
        if batch_logits.dim() == 1:
            batch_logits = batch_logits.unsqueeze(0)
        if batch_label.dim() > 1:
            batch_label = batch_label.squeeze()
        loss = self.criterion_loss_clf(batch_logits, batch_label)
        return {
            "loss": loss,
            "batch_logits": batch_logits.detach(),
            "batch_label": batch_label.detach(),
            "scores_attn": scores_attn.detach(),
        }

    # step_end template
    def step_end(self, outputs, label):
        # metrics
        if label!='train':
            self.step_metric(outputs["batch_logits"][:, 1], outputs["batch_label"], label)
        # loss
        if label!='test':
            self.log(f"{label}_loss", outputs['loss'].detach().item(), prog_bar=False, sync_dist=True)
