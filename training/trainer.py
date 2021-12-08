import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

from utils import AverageMeter


class Trainer:
    def __init__(
        self,
        dataloader_workers: int,
        device: str,
        epochs: int,
        learning_rate: float,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        pin_memory: bool,
        save_dir: str,
        train_batch_size: int,
        train_set: Dataset,
        valid_batch_size: int,
        valid_set: Dataset,
        evaluate_on_accuracy: bool = False
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=train_batch_size,
            num_workers=dataloader_workers,
            pin_memory=pin_memory,
            shuffle=False
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.train_loss = AverageMeter()
        self.evaluate_on_accuracy = evaluate_on_accuracy
        if evaluate_on_accuracy:
            self.best_valid_score = 0
        else:
            self.best_valid_score = float("inf")

    def train(self) -> None:
        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.train_loss.reset()

            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    self.optimizer.zero_grad()
                    data = {key: value.to(self.device) for key, value in data.items()}
                    output = self.model(**data)
                    loss = output.loss
                    loss.backward()
                    self.optimizer.step()
                    self.train_loss.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)

            if self.evaluate_on_accuracy:
                valid_accuracy = self.evaluate_accuracy(self.valid_loader)
                if valid_accuracy > self.best_valid_score:
                    print(
                        f"Validation accuracy improved from {self.best_valid_score:.4f} to {valid_accuracy:.4f}. Saving."
                    )
                    self.best_valid_score = valid_accuracy
                    self._save()
            else:
                valid_loss = self.evaluate(self.valid_loader)
                if valid_loss < self.best_valid_score:
                    print(
                        f"Validation loss decreased from {self.best_valid_score:.4f} to {valid_loss:.4f}. Saving.")
                    self.best_valid_score = valid_loss
                    self._save()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        eval_loss = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}
                output = self.model(**data)
                loss = output.loss
                eval_loss.update(loss.item(), self.valid_batch_size)
                tepoch.set_postfix({"valid_loss": eval_loss.avg})
                tepoch.update(1)
        return eval_loss.avg

    @torch.no_grad()
    def evaluate_accuracy(self, dataloader: DataLoader) -> float:
        self.model.eval()
        accuracy = AverageMeter()
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}
                output = self.model(**data)
                preds = torch.argmax(output.logits, dim=1)
                score = accuracy_score(data["labels"].cpu(), preds.cpu())
                accuracy.update(score, self.valid_batch_size)
                tepoch.set_postfix({"valid_acc": accuracy.avg})
                tepoch.update(1)
        return accuracy.avg

    def _save(self) -> None:
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)
