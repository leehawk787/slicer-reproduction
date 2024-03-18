A reproduction of Fabian David Schmidt's et. al. paper [SLICER: Sliced Fine-Tuning for Low-Resource Cross-Lingual Transfer for Named Entity Recognition] (https://aclanthology.org/2022.emnlp-main.740/) for the Multilingual - NLP Language in Summer 2023 @ University Würzburg. Joined work with @SimonServant and Frederik Pilz.

## Main Idea
The core idea is to "force" a different attention pattern in a transformers last layer: Dividing the embedding of each token into small dimension slices (like in multihead), making a prediction on each small slice and then linearly combining the results. \\ This is hoped to lead to less in-token attention and more between-token attention, which is shown to be beneficial for multilingual low-resource Named Entity Recognition tasks.

## Core Code
```

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

```

## Why our results are slightly different from the original

 * A different BIO schema (all-B-subwords -> B vs I)  used to tag subwords
 * FP16 vs FP32
 * An optional pooling layer before the classifier

We do not expect to complete the analysis 100%. 
