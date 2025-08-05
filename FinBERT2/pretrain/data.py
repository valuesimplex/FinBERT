import os
import random
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import DataCollatorForWholeWordMask

from utils import tensorize_batch
random.seed(42)


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        if os.path.isdir(data_dir):
            datasets = []
            for file in os.listdir(data_dir):
                print(f"Loading {file}")
                file = os.path.join(data_dir, file)
                datasets.append(self.load_dataset(file))
            self.dataset = concatenate_datasets(datasets)
        else:
            print(f"Loading {data_dir}")
            self.dataset = self.load_dataset(data_dir)

    def load_dataset(self, file):
        if file.endswith('.jsonl') or file.endswith('.json'):
            return load_dataset('json', data_files=file)['train']
        elif os.path.isdir(file):
            return Dataset.load_from_disk(file)
        else:
            raise NotImplementedError(f"Not support this file format:{file}")

    def __getitem__(self, item):
        return self.dataset[item]['text']

    def __len__(self):
        return len(self.dataset)


@dataclass
class mlmCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []

        for e in examples:
            e_trunc = self.tokenizer.encode(e, max_length=self.max_seq_length, truncation=True)
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            mask_set = []
            for _ in range(min(len(tokens), 128)):
                mask_set.append(self._whole_word_mask(tokens))

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)

        batch = {
            "input_ids": encoder_input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": encoder_labels_batch,
            # "examples":examples     #  run_mlm_getloss文件使用
        }

        return batch