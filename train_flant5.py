#!/usr/bin/env python
"""
FLAN-T5 for Social Isolation Detection
This script performs fine tuning of FLAN-T5-large across multiple healthcare sites for social isolation detection.
"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

import re
import json
from collections import Counter
import unicodedata
import pandas as pd
import numpy as np
import random
from datasets import Dataset

# Classifiction
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

def main():
    # Define valid labels
    valid_labels = ['no social reference', 'social isolation', 'social support']

    def clean_text(text):
        if not isinstance(text, str) or not text.strip():
            return ""
            
        text = text.upper()
        
        # Normalize weird spaces and symbols
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("�", " ")
        text = text.replace("UNABLE TO DISPLAY)", "UNABLE TO DISPLAY")
        text = re.sub(r"<OBX\.5\.\d+>", " ", text)
        text = re.sub(r"&lt;|&gt;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    # Function to set random seed for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Set random seed
    seed_value = 43
    set_seed(seed_value)

    # Clear GPU Memory
    torch.cuda.empty_cache()

    # Load data
    def load_json_file(file_path):
        all_data = []
    
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
    
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Unexpected format in file {file_path}, skipping.")
    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in {file_path}: {e}")
    
        return all_data

    # Load data 
    data = load_json_file('/data/reannotated_data.json')

    labelled_data, unlabelled_data = [], []

    for entry in data:
        text = entry["text"]
        span = entry.get("span","")
        if span == None:
            span = clean_text(text)
        else:
            span = clean_text(span)
        label = entry["label"]
        text = clean_text(text)
        
        if label in valid_labels and span:
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
    no_si = df[df['label'] == 'no social reference']
    no_si = no_si.drop_duplicates(subset=["text", "label"])
    others = df[df['label'] != 'no social reference']
    df_balanced = pd.concat([no_si, others], ignore_index=True)

    deduplicate_data = Dataset.from_pandas(df_balanced, preserve_index=False)

    # Split
    split_data = deduplicate_data.train_test_split(test_size=0.2, seed=seed_value)
    train_data = split_data['train']
    test_data = split_data['test']

    # Tokenizer
    model_name = '/llm_model_files/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    def preprocess_function(example):
        prompt = (
            "Classify the span based on the context into one of: "
            "['social isolation', 'no social reference', or 'social support'].\n"
            f"Text: {example['text']}\n"
            f"Span: {example['span']}\n\n"
            "Label:"
        )
    
        inputs = tokenizer(
            prompt,
            max_length=256,
            truncation=True,
            padding="max_length"
        )
    
        labels = tokenizer(
            text_target=example["label"],
            max_length=10,
            truncation=True,
            padding="max_length"
        )["input_ids"]
    
        inputs["labels"] = [
            token if token != tokenizer.pad_token_id else -100
            for token in labels
        ]
    
        return inputs

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

    # Evaluation
    def predict(text, span, model, tokenizer):
        prompt = (
            "Classify the span based on the context into one of: "
            "['social isolation', 'no social reference', or 'social support'].\n"
            f"Text: {text}\n"
            f"Span: {span}\n\n"
            "Label:"
        )
    
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(model.device)
    
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_length=20
            )
    
        return tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()
    
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Predictions
    true_labels, predicted_labels, spans = [], [], []

    for example in test_data:
        actual_label  = example.get("label", "unknown")
        text = example["text"]
        span = example["span"]
        predicted_label = predict(
            text, 
            span, 
            model, 
            tokenizer
        ).strip().lower()

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
