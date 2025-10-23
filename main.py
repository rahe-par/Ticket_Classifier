!pip install transformers datasets evaluate accelerate -q

import pandas as pd
import numpy as np
import re
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

# 1. LOAD DATASET
print("Loading multilingual customer support tickets dataset...")
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
df = pd.DataFrame(dataset["train"])

# Use relevant text fields only
df = df[['subject', 'body', 'queue']].dropna(subset=['body', 'queue'])
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')

# 2. CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["subject"] = df["subject"].apply(clean_text)
df["body"] = df["body"].apply(clean_text)
df["text"] = df["subject"] + " [SEP] " + df["body"]

# 3. ENCODE LABELS
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["queue"])
num_labels = len(label_encoder.classes_)
print(f"Detected {num_labels} label classes")

# 4. SPLIT DATA
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# 5. TOKENIZATION
MODEL_NAME = "xlm-roberta-base"  # multilingual backbone
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

train_dataset = {
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
}
test_dataset = {
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
}

import torch
class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["labels"])

train_ds = TicketDataset(train_dataset)
test_ds = TicketDataset(test_dataset)

# 6. METRICS
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {**acc, **f1}

# 7. MODEL
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",      
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.02,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# 9. TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# 10. TRAIN MODEL
trainer.train()

# 11. EVALUATE
metrics = trainer.evaluate()
print("Final Evaluation:", metrics)

# Save model for future inference
model.save_pretrained("xlm_roberta_tickets_90plus")
tokenizer.save_pretrained("xlm_roberta_tickets_90plus")

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\nEvaluating the fine-tuned model...")

# Step 1: Predict on test set using Hugging Face Trainer
test_preds_output = trainer.predict(test_ds)
logits = test_preds_output.predictions
y_pred_classes = np.argmax(logits, axis=1)
y_test_classes = np.array(test_labels)

# Step 2: Calculate and print metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Multilingual Transformer)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

!pip install -q -U google-genai

from google import genai

# --- Gemini API integration ---
import os
print("\nSetting up Gemini API for response generation...")
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = input("Paste your Gemini API key: ").strip()

    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Could not configure Gemini API: {e}")
    client = None
def generate_customer_reply(ticket_text, predicted_queue):
    if client is None:
        return "Gemini API not configured; skipping response generation."
    prompt = f"""
    You are a helpful, empathetic customer support assistant.
    A customer has submitted a ticket categorized as "{predicted_queue}".
    Message: "{ticket_text}"
    Write a brief, professional, warm acknowledgment note confirming receipt.
    """
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating Gemini response: {e}"


print("\n--- Demonstrating Full Pipeline: Classify and Reply ---")
# Select a random sample from the original test data
sample_idx = np.random.choice(len(test_texts))
sample_text = test_texts[sample_idx]

# Tokenize and predict (PyTorch path)
predict_input = tokenizer(sample_text, truncation=True, padding=True, return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predict_input = tokenizer(sample_text, truncation=True, padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    output = model(**predict_input).logits
with torch.no_grad():
    output = model(**predict_input).logits
prediction_value = np.argmax(output.cpu().numpy())
predicted_queue = label_encoder.inverse_transform([prediction_value])[0]

# Find original subject/body for Gemini
original_subject = df[df['text'] == sample_text]['subject'].values[0]
original_body = df[df['text'] == sample_text]['body'].values[0]
gemini_input_text = f"Subject: {original_subject}\n\nBody: {original_body}"

generated_reply = generate_customer_reply(gemini_input_text, predicted_queue)

print(f"\nOriginal Ticket Text (Combined for Model):\n\"{sample_text}\"")
print("-" * 30)
print(f"Predicted Queue: {predicted_queue}")
print("-" * 30)
print(f"Generated Reply (from Gemini):\n{generated_reply}")
print("-" * 30)
print("\nProject execution finished.")
