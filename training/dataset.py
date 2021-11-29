import datasets
import random
import pandas as pd
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple
import en_core_web_sm


class QGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: datasets.Dataset,
        max_length: int,
        pad_mask_id: int,
        tokenizer: AutoTokenizer
    ) -> None:
        self.data = pd.DataFrame(data)
        self.max_length = max_length
        self.pad_mask_id = pad_mask_id
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data.loc[index]
        input_ids, attention_mask = self._encode_text(item.text)
        labels, _ = self._encode_text(item.question)
        masked_labels = self._mask_label_padding(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": masked_labels
        }

    def _encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return (
            encoded_text["input_ids"].squeeze(),
            encoded_text["attention_mask"].squeeze()
        )

    def _mask_label_padding(self, labels: torch.Tensor) -> torch.Tensor:
        labels[labels == self.tokenizer.pad_token_id] = self.pad_mask_id
        return labels


class QAEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.Dataset, max_length: int, tokenizer: AutoTokenizer) -> None:
        self.data = pd.DataFrame(data)
        self.max_length = max_length
        self.transforms = [self.shuffle, self.corrupt]
        self.hf_tokenizer = tokenizer
        self.spacy_tokenizer = en_core_web_sm.load()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        question, answer = self.data.loc[index]
        label = random.choice([0, 1])
        if label == 0:
            question, answer = random.choice(self.transforms)(question, answer)
        encoded_data = self.hf_tokenizer(
            text=question,
            text_pair=answer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded_data["input_ids"].squeeze(),
            "attention_mask": encoded_data["attention_mask"].squeeze(),
            "token_type_ids": encoded_data["token_type_ids"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.int64)
        }

    def shuffle(self, question: str, answer: str) -> Tuple[str, str]:
        shuffled_answer = answer
        while shuffled_answer == answer:
            shuffled_answer = self.data.sample(1)['answer'].item()
        return question, shuffled_answer

    def corrupt(self, question: str, answer: str) -> Tuple[str, str]:
        doc = self.spacy_tokenizer(question)
        if len(doc.ents) > 1:
            # Replace all entities in the sentence with the same thing
            copy_ent = str(random.choice(doc.ents))
            for ent in doc.ents:
                question = question.replace(str(ent), copy_ent)
        elif len(doc.ents) == 1:
            # Replace the answer with an entity from the question
            answer = str(doc.ents[0])
        else:
            question, answer = self.shuffle(question, answer)
        return question, answer
