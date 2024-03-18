import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics.functional
from torch import argmax
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from transformers import AutoModel
from torchmetrics.functional.classification import f1_score, accuracy, precision, recall
from torch.nn.functional import cross_entropy

from .slicer_nn_modules import SlicerRobertaNER, RobertaNER

label_dict = {0: 'O', 1:  'B-PER', 2: 'I-PER',3: 'B-ORG',4: 'I-ORG',5: 'B-LOC', 6: 'I-LOC',7: 'B-MISC',8: 'I-MISC'}
def lookup_table(label):
    return label_dict[label]

class SlicerLightningModule(pl.LightningModule):
    def __init__(self,
                test_datasets_names,
                 n_labels = 7,
                 hidden_size = 768, 
                 learning_rate = 2e-5, 
                 weight_decay = 0.05,
                 is_slicer = True,
                 slice_size = 4
                 ):
        # Save HP to checkpoints
        super().__init__()
        self.n_labels = n_labels
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weigt_decay = weight_decay
        self.is_slicer = is_slicer
        self.slice_size = slice_size
        self.num_slices = int(self.hidden_size/self.slice_size)
        self.test_datasets_names  = test_datasets_names
        self.save_hyperparameters()

        # Init model
        if (self.is_slicer):
            self.model = SlicerRobertaNER(n_labels=self.n_labels, slice_size=slice_size, num_slices=self.num_slices)
        else:
            self.model = RobertaNER(n_labels=self.n_labels)


    def __default_step(self, batch, batch_idx):
        x, labels = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, batch['labels']


        logits = self.model(x).view(-1, self.n_labels)  # combine batch and sequence (and num_slices) into one dim
        labels = labels.to(logits.device).view(-1)  # combine batch and sequence into one dim

        # length of logits is different depending on SLICER/STANDARD
        if (self.is_slicer):
            labels = labels.repeat_interleave(self.num_slices)

        loss = cross_entropy(logits, labels, ignore_index=-100)

        return logits, labels, loss




    def training_step(self, batch, batch_idx):
        _,_, loss = self.__default_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, loss = self.__default_step(batch, batch_idx)
        micro_f1, f1, precision, recall, accuracy = self.computeMetrics(logits, labels)

        self.log("val_f1", micro_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.logMetrics(loss, f1, recall, precision, accuracy, "val")


    def test_step(self, batch, batch_idx, dataloader_idx):

        logits, labels, loss = self.__default_step(batch, batch_idx)
        micro_f1, f1, precision, recall, accuracy = self.computeMetrics(logits, labels)

        dataset_name = self.test_datasets_names[dataloader_idx]
        self.log(f"{dataset_name}_test_f1", micro_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.logMetrics(loss, f1, recall, precision, accuracy, f"{dataset_name}_test")


    def computeMetrics(self, logits, labels):
        # 7-er vector giving scores for each class individually
        micro_f1 = f1_score(preds=logits, target=labels,
                                 task='multiclass', num_classes=self.n_labels, ignore_index=-100, average='micro', multidim_average='global')
        f1 = f1_score(preds=logits, target=labels,
                           task='multiclass', num_classes=self.n_labels, ignore_index=-100, average='none', multidim_average='global')
        p = precision(preds=logits, target=labels,
                                   task='multiclass', num_classes=self.n_labels, ignore_index=-100, average='none', multidim_average='global')
        r = recall(preds=logits, target=labels,
                             task='multiclass', num_classes=self.n_labels, ignore_index=-100, average='none',multidim_average='global')
        a = accuracy(preds=logits, target=labels,
                                 task='multiclass', num_classes=self.n_labels, ignore_index=-100, average='none',multidim_average='global')

        return micro_f1, f1, p, r, a

    def logMetrics(self, loss, f1, recall, precision, accuracy, stage : str):
        #9er vector giving scores for each class individually
        dict = {f"{stage}_loss": loss}
        for i in range(self.n_labels):
            dict[f"{lookup_table(i)}_{stage}_recall"] = recall[i]
            dict[f"{lookup_table(i)}_{stage}_precision"] = precision[i]
            dict[f"{lookup_table(i)}_{stage}_f1"] = f1[i]
            dict[f"{lookup_table(i)}_{stage}_accuracy"] = accuracy[i]
        self.log_dict(dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)



    # def predict_step(self, batch, batch_idx):

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 
                                lr=self.learning_rate, 
                                weight_decay=self.weigt_decay)

