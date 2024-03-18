import os
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base", cache_dir=os.getenv("CACHE_DIR"))

# Align label with subtokens generated through tokenizer
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = TOKENIZER(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

'''
taken from the SPLICER repository: Label indices larger than 6 are remapped to 0.
'''
def remap_to_wikiann_labels(examples: dict) -> dict:
    examples["ner_tags"] = [
        [tag if tag < 7 else 0 for tag in instance] for instance in examples["ner_tags"]
    ]
    return examples

class MonolingualNERDataSet:
    def __init__(self,
                 name="conll2003",
                 split="train",
                 languages=None) -> None:
        super().__init__()
        if languages:
            self.dataset = datasets.concatenate_datasets(list(load_dataset(name, lang, cache_dir=os.getenv("CACHE_DIR"), split=split) for lang in languages))
        else:
            self.dataset = load_dataset(name, cache_dir=os.getenv("CACHE_DIR"), split=split)
        self.dataset_name = name
        self.tokenized_datasets = self.dataset.map(remap_to_wikiann_labels,batched=True)
        self.tokenized_datasets = self.tokenized_datasets.map(tokenize_and_align_labels, batched=True).map(batched=True, remove_columns=self.dataset.column_names)
        self.tokenized_datasets.set_format(type="torch", columns=["input_ids","attention_mask", "labels"])

class MultilingualNERDataSet:
    def __init__(self,
                 name="masakhaner",
                 languages=["amh", "hau", "ibo", "kin", "luo", "pcm", "swa", "wol", "yor"],
                 split="test") -> None:
        super().__init__()
        self.dataset_names = languages
        self.datasets = list(load_dataset(name, lang, split=split, cache_dir=os.getenv("CACHE_DIR")) for lang in languages)
        self.tokenized_datasets = [dataset.map(remap_to_wikiann_labels,batched=True).map(tokenize_and_align_labels, batched=True).map(batched=True, remove_columns=self.datasets[0].column_names) for dataset in self.datasets]
        for dataset in self.tokenized_datasets:
            dataset.set_format(type="torch", columns=["input_ids","attention_mask", "labels"])

