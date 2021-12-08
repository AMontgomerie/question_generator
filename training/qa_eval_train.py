import argparse
import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy

from dataset import QAEvalDataset
from trainer import Trainer

spacy.prefer_gpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--qa_eval_model", type=str, default="bert-base-cased")
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./bert-base-cased-qa-evaluator")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.qa_eval_model)
    dataset = datasets.load_dataset("iarfmoose/qa_evaluator")
    train_set = QAEvalDataset(dataset["train"], args.max_length, tokenizer)
    valid_set = QAEvalDataset(dataset["validation"], args.max_length, tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.qa_eval_model)
    trainer = Trainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        train_batch_size=args.train_batch_size,
        train_set=train_set,
        valid_batch_size=args.valid_batch_size,
        valid_set=valid_set,
        evaluate_on_accuracy=True
    )
    trainer.train()
