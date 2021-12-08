import argparse
import datasets
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from dataset import QGDataset
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--qg_model", type=str, default="t5-base")
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./t5-base-question-generator")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    return parser.parse_args()


def get_tokenizer(checkpoint: str) -> T5Tokenizer:
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )
    return tokenizer


def get_model(checkpoint: str, device: str, tokenizer: T5Tokenizer) -> T5ForConditionalGeneration:
    config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = T5ForConditionalGeneration(config).from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model


if __name__ == "__main__":
    args = parse_args()
    tokenizer = get_tokenizer(args.qg_model)
    dataset = datasets.load_dataset("iarfmoose/question_generator")
    train_set = QGDataset(dataset["train"], args.max_length, args.pad_mask_id, tokenizer)
    valid_set = QGDataset(dataset["validation"], args.max_length, args.pad_mask_id, tokenizer)
    model = get_model(args.qg_model, args.device, tokenizer)
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
        valid_set=valid_set
    )
    trainer.train()
