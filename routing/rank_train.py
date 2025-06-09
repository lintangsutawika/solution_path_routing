import sys
import random
import argparse

import numpy as np

import torch
import torch.nn.functional as F

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
    )

def compute_loss_func(outputs, labels, **kwargs):
    
    predictions = outputs[0]

    bs, *_ = predictions.shape
    # label example [1, 4, 3, 2, 0]
    # Lower mean higher rank
    def _loss_fn(predictions, labels):
        m = len(labels)
        y = m - labels
        gamma = torch.FloatTensor([random.random() for _ in range(len(labels))]).to(predictions.device)
        Z_phi = sum([2**y[i] - gamma[i] for i in range(len(labels))])

        loss = torch.tensor(0.0, device=predictions.device)
        phi = (2**y - gamma) / Z_phi
        rho = predictions/sum(predictions)

        loss = F.cross_entropy(phi, rho)
        return loss/m

    loss = torch.tensor(0.0, device=predictions.device)
    for i in range(bs):
        loss += _loss_fn(predictions[i], labels[i])

    return loss/bs


def compute_metrics(pred):

    for prediction, label in zip(pred.predictions, pred.label_ids):

        y = [len(label) - i for i in label]

        prediction = prediction.tolist()
        pi_f = [prediction.index(i) + 1 for i in sorted(prediction)]
        dcg_f = sum([y[i]/np.log2(pi_f[i]+1) for i in range(len(label))])

        label = label.tolist()
        pi_y = [l + 1 for l in  label]
        dcg_y = sum([y[i]/np.log2(pi_y[i]+1) for i in range(len(label))])

        # Calculate NDCG
        ndcg = dcg_f / dcg_y

    return {
        'NDCG': ndcg,
    }

def main(args):
    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "test": "data/test.jsonl"
        }
    )

    # Load tokenizer and model
    model_name = args.model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=2048, truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Prepare datasets for training
    train_dataset = tokenized_datasets["train"] # .shuffle(seed=args.seed).select(range(args.train_subset_size))
    test_dataset = tokenized_datasets["test"] # .shuffle(seed=args.seed).select(range(args.test_subset_size))

    print(train_dataset)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss_func,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Pretrained model name")
    parser.add_argument("--num_labels", type=int, default=10, help="Number of labels for classification")
    parser.add_argument("--train_subset_size", type=int, default=2000, help="Subset size for training")
    parser.add_argument("--test_subset_size", type=int, default=500, help="Subset size for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for output results")
    parser.add_argument("--save_model_dir", type=str, default="./trained_model", help="Directory to save the trained model")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    args = parser.parse_args()
    main(args)