import torch
from torch import nn
from transformers import AutoModel


class SlicerRobertaNER(nn.Module):
    def __init__(self,
                 n_labels=9,
                 slice_size=4,
                 hidden_size=768,
                 num_slices=192):
        # Save HP to checkpoints
        super().__init__()
        self.n_labels = n_labels
        self.hidden_size = hidden_size
        self.slice_size = slice_size
        self.num_slices = num_slices

        # Init model
        self.base = AutoModel.from_pretrained("xlm-roberta-base")
        linear = nn.Linear(self.hidden_size, self.n_labels)

        self.classifier_weights = linear.weight.data.T.reshape((self.num_slices, self.slice_size, self.n_labels))
        self.classifier_bias = linear.bias

        #self.classifier_weights.requires_grad = True
        #self.classifier_bias.requires_grad = True
        self.classifier_weights = torch.nn.Parameter(self.classifier_weights, requires_grad=True)
        self.classifier_bias = torch.nn.Parameter(self.classifier_bias, requires_grad=True)

        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        classified_outputs = self.base(**x)
        logits = classified_outputs[0]



        ####
        batch_size, sequence_length, hidden = logits.shape
        ####


        ####
        num_slices, slice_size, n_labels = self.classifier_weights.shape
        ####

        # we slice along the hidden dimension in num_slices slides
        sliced_outputs = logits.reshape((batch_size * sequence_length, self.num_slices, self.slice_size))
        # sliced_classifier= weight_matrix.reshape((self.num_slices, self.slice_size, self.n_labels)).to(sliced_outputs.device)
        self.classifier_weights = self.classifier_weights.to(sliced_outputs.device)

        # and combine them:
        # target shape
        # classified_outputs = torch.zeros((batch_size*sequence_length, self.num_slices, self.n_labels)).to(sliced_outputs.device)
        # combination
        # for i in range(batch_size*sequence_length):
        #    for j in range(self.num_slices):
        #        for kk in range(self.slice_size):
        #            for l in range(self.n_labels):
        #                classified_outputs[i, j, l] += sliced_outputs[i, j, kk] * sliced_classifier[j, kk, l]

        classified_outputs = torch.einsum("ndk, dkl->ndl", sliced_outputs, self.classifier_weights).reshape((-1,self.n_labels)) + self.classifier_bias

        # reshape to be compatible with later operations

        return classified_outputs


class RobertaNERAdvanced(nn.Module):
    def __init__(self,
                 n_labels=9,
                 hidden_size=384):
        # Save HP to checkpoints
        super().__init__()
        self.n_labels = n_labels
        self.hidden_size = hidden_size
        # Init model

        self.base = AutoModel.from_pretrained("xlm-roberta-base")
        self.dense = nn.Linear(768, hidden_size)
        # self.dropout = nn.Dropout(p=0.2)
        self.classification_head = nn.Linear(hidden_size, n_labels)

    def forward(self, x):
        output = self.base(**x)
        logits = output[0]
        logits = self.dense(logits)
        # logits = self.dropout(logits)
        logits = self.classification_head(logits)
        return logits


class RobertaNER(nn.Module):
    def __init__(self,
                 n_labels=9):
        # Save HP to checkpoints
        super().__init__()
        self.n_labels = n_labels
        # Init model

        self.base = AutoModel.from_pretrained("xlm-roberta-base")
        self.classification_head = nn.Linear(768, n_labels)

    def forward(self, x):
        output = self.base(**x)
        logits = output[0]
        logits = self.classification_head(logits)
        return logits