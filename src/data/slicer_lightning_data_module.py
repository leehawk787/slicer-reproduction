import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

from .data_set.splicer_ner_dataset import MonolingualNERDataSet, MultilingualNERDataSet

_ENGLISH = [
    "en"
]

_MASAKHANER_LANGS = ["amh", "hau", "ibo", "kin", "luo", "pcm", "swa", "wol", "yor"]

class SlicerLightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=2, tokenizer_checkpoint="xlm-roberta-base"):
        super().__init__()
        self.model_checkpoint = tokenizer_checkpoint
        self.batch_size = batch_size
        self.test_datasets_names  = None
        self.data_collator = None
        self.tokenizer = None
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage="train_conll"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="pt")

        if stage == "train_conll":
            # Load data for training
            dataset = MonolingualNERDataSet(name="conll2003", split="train")
            self.train_dataset = dataset.tokenized_datasets

            # Load data for validation
            datasets = MonolingualNERDataSet(name="conll2003", split="validation")
            self.validation_dataset = datasets.tokenized_datasets        

        if stage == "train_wikiann":
            # Load data for training
            dataset = MonolingualNERDataSet(name="wikiann", split="train", languages=_ENGLISH)
            self.train_dataset = dataset.tokenized_datasets

            # Load data for validation
            datasets = MonolingualNERDataSet(name="wikiann", split="validation", languages=_ENGLISH)
            self.validation_dataset = datasets.tokenized_datasets

        masakhaner = MultilingualNERDataSet("masakhaner", _MASAKHANER_LANGS, split="test")
        self.test_datasets_names = masakhaner.dataset_names
        self.test_datasets = masakhaner.tokenized_datasets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.data_collator, 
            # num_workers=self.num_workers
            )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.data_collator,
            # num_workers=self.num_workers
            )

    def test_dataloader(self):
        dataloaders = {}
        for index in range(len(self.test_datasets_names)):
            dataloader = DataLoader(
                self.test_datasets[index], 
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.data_collator)
            name = self.test_datasets_names[index]
            dataloaders[name] = dataloader
        return dataloaders
    # HINT: Evaluating on multiple dataloaders (https://lightning.ai/docs/pytorch/LTS/guides/data.html)

