import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import pipeline
from pathlib import Path
from datasets import Dataset
import numpy as np
from evaluate import load

import data_preprocess
df = data_preprocess.parse_xml()

os.environ["HF_HOME"] = "C:\\temp\\huggingface"

path = Path(__file__).resolve().parent.parent 
data_path = path.joinpath('data')
result_path = path.joinpath('result')

metric = load("accuracy")  

label_encoder = LabelEncoder()
df["tags"] = label_encoder.fit_transform(df["tags"])  

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1) 
    return metric.compute(predictions=predictions, references=labels)

def train_model(df):
    df['text'] = df['title'] + " " + df['body']

    train_texts, val_texts, train_tags, val_tags = train_test_split(df['text'], df['tags'], test_size=0.2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_enc = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
    val_enc = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)
    
    train_enc["labels"] = list(map(int, train_tags))
    val_enc["labels"] = list(map(int, val_tags)) 

    train_dataset = Dataset.from_dict(train_enc)
    val_dataset = Dataset.from_dict(val_enc)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(set(df["tags"])))

    training_args = TrainingArguments(output_dir=str(result_path), per_device_train_batch_size=8, num_train_epochs=3)
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

# trained_model = train_model(df)  
