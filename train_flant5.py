#!/usr/bin/env python
"""
FLAN-T5 for Social Isolation Detection
This script performs fine tuning of FLAN-T5-large across multiple healthcare sites for social isolation detection.
"""
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

import os
import re
import string
import json
from collections import defaultdict, Counter
import unicodedata
import pandas as pd
import numpy as np
import random
from datasets import Dataset, DatasetDict


def main():
    # Set random seed
    seed_value = 43
    set_seed(seed_value)

    # Clear GPU Memory
    torch.cuda.empty_cache()

    # Load data 
    data = load_json_file('/data/reannotated_data.json')

    labelled_data, unlabelled_data = [], []

    for entry in data:
        text = entry["text"]
        span = entry.get("span","")
        if span == None:
            span = text
        label = entry["label"]
        processed_text = clean_text(text)
        span_text = clean_text(span)
        if label in valid_labels and span_text:
            labelled_data.append({"text":text, "span": span, "label": label})
        else:
            unlabelled_data.append({"text":text, "span": span, "label": label})

    class_count_labelled_dataset = Counter([item["label"] for item in labelled_data])
    total_sample_labelled_dataset = sum(class_count_labelled_dataset.values())
    max_count_labelled_dataset = max(class_count_labelled_dataset.values())

    print(f"Total samples in labelled dataset: {total_sample_labelled_dataset}")
    print(f"Total class count in labelled dataset:\n {class_count_labelled_dataset}")
    print(f"Max count of classes in labelled dataset:\n {max_count_labelled_dataset}")

    # Convert to dataset
    formatted_labelled_data = Dataset.from_list(labelled_data)

    # Remove duplicates
    df = formatted_labelled_data.to_pandas()
    no_si = df[df['label'] == 'no social isolation']
    no_si = no_si.drop_duplicates(subset=["text", "label"])
    others = df[df['label'] != 'no social isolation']
    df_balanced = pd.concat([no_si, others], ignore_index=True)

    deduplicate_data = Dataset.from_pandas(df_balanced, preserve_index=False)

    # Split
    split_data = deduplicate_data.train_test_split(test_size=0.2, seed=seed_value)
    train_data = split_data['train']
    test_data = split_data['test']

    # Tokenizer
    model_name = '/llm_model_files/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Tokenize
    tokenized_train_data = train_data.map(preprocess_function, remove_columns=["text", "span", "label"])
    tokenized_test_data = test_data.map(preprocess_function, remove_columns=["text", "span", "label"])

    # Model
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Training args
    training_args = TrainingArguments(
        output_dir="/finetuned_models/flan_t5_unique_spans_context",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.05,
        save_strategy="epoch",
        logging_dir="/fine_tune_logs",
        logging_steps=10,
        push_to_hub=False,
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=seed_value,
        dataloader_num_workers=8,
        fp16=torch.cuda.is_available(),
        warmup_steps=100,
        eval_steps=500,
        save_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Predictions
    true_labels, predicted_labels, spans = [], [], []

    for example in test_data:
        span = example["text"]
        actual_label  = example.get("label", "unknown")
        predicted_label = predict(example["text"], model, tokenizer)

        spans.append(span)
        true_labels.append(actual_label)
        predicted_labels.append(predicted_label)

        print(f"Text span: {span}")
        print(f"Actual label: {actual_label}")
        print(f"Predicted label: {predicted_label}")
        print("-" * 50)

    print("\n Classification Report:\n")
    print(classification_report(true_labels, predicted_labels, target_names=valid_labels))

    y_true = [valid_labels.index(label) for label in true_labels]
    y_pred = [valid_labels.index(label) if label in valid_labels else 0 for label in predicted_labels]
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"F-Score: {f1:.4f}")

    # Save model
    save_dir = '/manual_save/flan_t5_unique_spans_w_context'
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()