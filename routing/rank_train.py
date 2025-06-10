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

    bs, m = labels.shape

    ## Relevance from 1 and decreasing
    # y = (m - labels)/m
    # Relevance follows a power law
    y = 1 / (labels + 1)
    gamma = torch.rand(y.shape).to(y.device)

    # From Literature
    phi = (2**y-gamma)/(2**y-gamma).sum(dim=-1, keepdim=True)
    rho = outputs.logits
    loss = F.cross_entropy(rho, phi, reduction='mean')/m

    return loss


def compute_metrics(pred):

    topk = 3
    ndcg = 0.0
    mrr = 0.0
    bs = 0
    for prediction, label in zip(pred.predictions, pred.label_ids):
        # mrr = sum([1/(pi_f[i]+1) for i in range(m) if y[i] > 0]) / m

        m = len(label)
        # Relevance of 1 for top 3 items, 0 for others
        y = (label < topk).astype(np.float32)
        # y = (m - label)/m

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.tolist()

        sorted_prediction = sorted(prediction, reverse=True)
        pi_f = np.asarray([prediction.tolist().index(x) + 1 for x in sorted_prediction])
        dcg_f = sum([(2**y[i]-1)/np.log2(pi_f[i]+1) for i in range(m)])

        mrr += 1/pi_f[label.tolist().index(0)]

        if isinstance(label, torch.Tensor):
            label = label.tolist()

        pi_y = np.asarray([l + 1 for l in label])
        dcg_y = sum([(2**y[i]-1)/np.log2(pi_y[i]+1) for i in range(m)])

        # Calculate NDCG
        ndcg += dcg_f / dcg_y
        bs += 1

    return {
        'NDCG': ndcg/bs,
        'MRR': mrr/bs,
        'NDCG+MRR': (ndcg + mrr)/2/bs,
    }

def main(args):
    # Load dataset
    dataset = load_dataset("json", data_dir=args.data_dir)
    # dataset = load_dataset("Yelp/yelp_review_full")

    # Load tokenizer and model
    model_name = args.model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id=tokenizer.pad_token_id

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=2048, truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Prepare datasets for training
    train_dataset = tokenized_datasets["train"].shuffle(seed=args.seed).select(range(args.train_subset_size))
    test_dataset = tokenized_datasets["test"].shuffle(seed=args.seed).select(range(args.test_subset_size))

    print(train_dataset)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        # eval_steps=32,
        eval_steps=8,
        save_steps=32,
        # save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        eval_on_start=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        # metric_for_best_model="NDCG",
        metric_for_best_model="NDCG+MRR",
        greater_is_better=True,
        # use_liger_kernel=True,
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Pretrained model name")
    parser.add_argument("--num_labels", type=int, default=10, help="Number of labels for classification")
    parser.add_argument("--train_subset_size", type=int, default=8192, help="Subset size for training")
    parser.add_argument("--test_subset_size", type=int, default=128, help="Subset size for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--output_dir", type=str, default="./model_checkpoints/", help="Directory for output results")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    args = parser.parse_args()
    main(args)