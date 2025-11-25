# encoding: utf-8
import os
from typing import Optional
from pytorch_lightning import LightningDataModule
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from life2lang.utils.span_corruption import protein_span_corruption


class ProteinSequenceDataModule(LightningDataModule):
    """
    DataModule for protein sequences.
    """
    def __init__(
        self,
        dataset_dir: str,
        tokenizer: AutoTokenizer,
        train_file_id: str = 'train.csv',
        val_file_id: str = 'validation.csv',
        protein_sequence_column: str = 'sequence',
        max_seq_len: int = 256,
        noise_density: float = 0.15,
        mean_span_length: int = 3,
        test_file_id: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.protein_sequence_column = protein_sequence_column
        self.train_file_id = train_file_id
        self.val_file_id = val_file_id
        self.max_seq_len = max_seq_len
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.test_file_id = test_file_id
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_file_path = os.path.join(self.dataset_dir, self.train_file_id)
            self.train_dataset = self._prepare(
                Dataset.from_csv(train_file_path) # type: ignore
            )

            valid_file_path = os.path.join(self.dataset_dir, self.val_file_id)
            self.valid_dataset = self._prepare(
                ds=Dataset.from_csv(valid_file_path) # type: ignore
            )
                

        if stage == 'test' or stage is None:
            if self.test_file_id is not None:
                test_file_path = os.path.join(self.dataset_dir, self.test_file_id) # type: ignore
                if not os.path.exists(test_file_path):
                    raise FileNotFoundError(f"Test file {test_file_path} does not exist.")
            else:
                self.test_dataset = self._prepare(
                    ds=Dataset.from_csv(os.path.join(self.dataset_dir, self.val_file_id)) # type: ignore
                )


    def _prepare(self, ds: Dataset):
        return ds.map(
            lambda x:
                protein_span_corruption(
                    x[self.protein_sequence_column],
                    mean_span_length=self.mean_span_length,
                    noise_density=self.noise_density
            ),
            batch_size=512,
            batched=True
        )

    def _format_protein(self, sequence: str) -> str:
        return f'[seq] {sequence} [/seq]'

    def _data_collator(self, batch):
        input_text = [self._format_protein(r['input_text']) for r in batch]
        target_text = [self._format_protein(r['target_text']) for r in batch]
        
        batch = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ) # type: ignore

        labels = self.tokenizer(
            target_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ) # type: ignore
        
        batch['labels'] = labels['input_ids']
        
        return batch

    def train_dataloader(self):        
        return DataLoader(
            self.train_dataset, # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._data_collator,
        )
