import sys
import random
import argparse

from functools import partial
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

from allrank.models.losses import lambdaLoss
from pytorchltr.loss import LambdaARPLoss1, LambdaNDCGLoss1, LambdaNDCGLoss2
from pytorchltr.evaluation import ndcg

# def compute_loss_func(outputs, labels, **kwargs):

#     bs, m = labels.shape

#     ## Relevance from 1 and decreasing
#     # y = (m - labels)/m
#     # Relevance follows a power law
#     y = 1 / (labels + 1)
#     gamma = torch.rand(y.shape).to(y.device)

#     # From Literature
#     phi = (2**y-gamma)/(2**y-gamma).sum(dim=-1, keepdim=True)
#     rho = outputs.logits
#     loss = F.cross_entropy(rho, phi, reduction='mean')/m

#     return loss

def custom_loss(rho, relevance, *args, **kwargs):

    bs, m = relevance.shape

    ## Relevance from 1 and decreasing
    # y = (m - labels)/m
    # Relevance follows a power law
    # y = 1 / (labels + 1)
    y = relevance
    gamma = torch.rand(y.shape).to(y.device)

    # From Literature
    phi = (2**y-gamma)/(2**y-gamma).sum(dim=-1, keepdim=True)
    loss = F.cross_entropy(rho, phi, reduction='mean')

    return loss

# loss_fn = custom_loss
# loss_fn = LambdaARPLoss1()
loss_fn = partial(lambdaLoss, weighing_scheme="ndcgLoss2PP_scheme", mu=5, reduction='sum')
# loss_fn = LambdaNDCGLoss2()

def rank_to_relevance(rank, score=None, use_softmax=False, **kwargs):

    # Create a tensor of power law scores: 1 / (rank + 1)
    if score is None:
        score = 1.0 / (torch.arange(rank.size(1)) + 1.0)  # shape [10]
    # score = score.to(rank.device)  # Ensure it is on the same device as rank

    # # Linear score
    # score = torch.FloatTensor(sorted(list(range(rank.size(1))), reverse=True))
    # score = torch.FloatTensor(sorted(list(range(rank.size(1))), reverse=True))/ rank.size(1)  # shape [10]
    score = score.to(rank.device)  # Ensure it is on the same device as rank

    # Initialize empty tensor for relevance scores
    relevance = torch.zeros_like(rank, dtype=torch.float).to(rank.device)  # shape [bs, 10]

    # Fill relevance: for each row, assign scores according to x's rank
    for i in range(rank.size(0)):
        relevance[i][rank[i]] = score

    if use_softmax:
        return torch.softmax(relevance, dim=1)
    else:
        return relevance

rank_type = {
    "power": {"score": 1.0 / (torch.arange(10) + 1.0)},
    "power_softmax": {"score": 1.0 / (torch.arange(10) + 1.0), "use_softmax": True},
    "linear": {"score": torch.FloatTensor(sorted(list(range(10)), reverse=True))/10.0},
    "linear_softmax": {"score": torch.FloatTensor(sorted(list(range(10)), reverse=True))/10.0, "use_softmax": True},
    "softmax": {"score": torch.FloatTensor(sorted(list(range(10)), reverse=True)), "use_softmax": True},
    "top3": {"score": torch.tensor([1,1,1,0,0,0,0,0,0,0], dtype=torch.float)},
}


def main(args):
    # Load dataset
    dataset = load_dataset("json", data_dir=args.data_dir)

    # Load tokenizer and model
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels)
    model.config.pad_token_id=tokenizer.pad_token_id

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=2048, truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("Tokenized datasets:", tokenized_datasets)
    # Prepare datasets for training
    train_dataset = tokenized_datasets["train"].shuffle(seed=args.seed) #.select(range(args.train_subset_size))
    validation_dataset = tokenized_datasets["validation"].shuffle(seed=args.seed) #.select(range(args.test_subset_size))

    print("Using Rank Type:", args.rank_type)

    rank_fn = partial(rank_to_relevance, **rank_type[args.rank_type])
    def compute_loss_func(outputs, labels, ce_lambda=0.5, **kwargs):

        bs, n = labels.shape
        predictions = outputs.logits
        relevance = rank_fn(labels)
        one_hot = F.one_hot(labels[:, 0], num_classes=n).float()

        loss = (1 - ce_lambda) * loss_fn(predictions, relevance) + \
             ce_lambda * F.cross_entropy(predictions, one_hot, reduction='sum')

        # loss = loss_fn(outputs.logits, relevance, n_size).mean()
        return loss

    def compute_metrics(pred):

        predictions, labels  = pred.predictions, pred.label_ids
        predictions = torch.tensor(predictions).to(labels.device)
        labels = torch.tensor(labels).to(predictions.device)
        bs, n = labels.shape
        n_size = torch.tensor([n]*bs)

        # score = torch.tensor([9,8,7,6,5,4,3,2,1,0], dtype=torch.float)
        # score = torch.tensor([5,4,3,2,1,0,0,0,0,0], dtype=torch.float)
        score = torch.tensor([1,1,1,0,0,0,0,0,0,0], dtype=torch.float)
        relevance = torch.zeros_like(labels, dtype=torch.float)
        # relevance = rank_fn(labels)

        for i in range(labels.size(0)):
            relevance[i][labels[i]] = score

        top_prediction = predictions.argmax(axis=1)
        top_label = labels[:, :1]
        
        precision = torch.zeros((1, 10), dtype=torch.float).to(predictions.device)
        recall = torch.zeros((1, 10), dtype=torch.float).to(predictions.device)
        for _class in range(10):
            true_positive = ((top_prediction == _class) & (top_label.squeeze() == _class)).float().sum()
            predicted_positive = (top_prediction == _class).float().sum()
            actual_positive = (top_label.squeeze() == _class).float().sum()

            precision[0, _class] = true_positive / predicted_positive if predicted_positive > 0 else 0.0
            recall[0, _class] = true_positive / actual_positive if actual_positive > 0 else 0.0

        precision = precision.mean().unsqueeze(0)
        recall = recall.mean().unsqueeze(0)
        f1_score = 2 * (precision * recall) / (precision + recall)
        # accuracy = (top_prediction == top_label.squeeze()).float().mean().item()

        ndcg_1 = ndcg(predictions, relevance, n_size, k=1, exp=False).mean().item()
        ndcg_2 = ndcg(predictions, relevance, n_size, k=2, exp=False).mean().item()
        ndcg_3 = ndcg(predictions, relevance, n_size, k=3, exp=False).mean().item()
        # ndcg_4 = ndcg(predictions, relevance, n_size, k=4, exp=False).mean().item()
        # ndcg_10 = ndcg(predictions, relevance, n_size, k=10, exp=False).mean().item()

        return {
            'F1': f1_score,
            'NDCG@1': ndcg_1,
            'NDCG@2': ndcg_2,
            'NDCG@3': ndcg_3,
            'Metric': (f1_score + ndcg_1 * 0.5 + ndcg_2 * 0.3 + ndcg_3 * 0.2) / 2,
        }

    # Define training arguments
    training_args = TrainingArguments(
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="default",
        output_dir=args.output_dir,
        eval_strategy="steps",
        max_steps=5_000,
        eval_steps=128,
        save_strategy="steps",
        save_steps=128,
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
        metric_for_best_model="Metric",
        greater_is_better=True,
        bf16=True,
        use_liger_kernel=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        compute_loss_func=partial(compute_loss_func, ce_lambda=args.ce_lambda),
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # if args.use_gcs:
    #     import fsspec
    #     from yeval.utils import simple_parse_args_string

    #     fs = fsspec.filesystem(
    #         "gcs",
    #         **simple_parse_args_string(args.file_system_kwargs) if args.file_system_kwargs else {}
    #     )

    #     fs.makedirs(run_path, exist_ok=True)
    #     result_file = os.path.join(run_path, "result.json")
    #     fs.put(args.output_dir, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument("--rank_type", type=str, default=None, help="Directory containing the dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Pretrained model name")
    parser.add_argument("--num_labels", type=int, default=10, help="Number of labels for classification")
    parser.add_argument("--train_subset_size", type=int, default=8192, help="Subset size for training")
    parser.add_argument("--test_subset_size", type=int, default=128, help="Subset size for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--output_dir", type=str, default="./model_checkpoints/", help="Directory for output results")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--ce_lambda", type=float, default=0.5, help="Lambda parameter for loss function")

    args = parser.parse_args()
    main(args)
