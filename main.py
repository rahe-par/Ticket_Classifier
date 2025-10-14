!pip install datasets tensorflow scikit-learn pandas numpy google-generativeai matplotlib seaborn transformers

# STEP 1: IMPORT LIBRARIES

import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import google.generativeai as genai
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, create_optimizer

# STEP 2: LOAD AND PREPARE THE DATASET

print("Loading dataset from Hugging Face...")
try:
    ds = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = pd.DataFrame(ds['train'])
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Failed to load dataset. Error: {e}")
    df = pd.DataFrame(data)

print("\n--- Original Data Sample ---")
print(df.head())

df = df[['subject', 'body', 'queue']]
df.dropna(subset=['body', 'queue'], inplace=True) # Only drop if body or queue is missing
df['subject'].fillna('', inplace=True) # Fill missing subjects with an empty string

# Basic text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

df['subject'] = df['subject'].apply(clean_text)
df['body'] = df['body'].apply(clean_text)

# COMBINE SUBJECT AND BODY
print("\nCombining 'subject' and 'body' fields for more context...")
df['combined_text'] = df['subject'] + ' [SEP] ' + df['body']

df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the dataset

# STEP 3: LABEL ENCODING

print("\nEncoding target labels...")
label_encoder = LabelEncoder()
df['queue_encoded'] = label_encoder.fit_transform(df['queue'])
num_classes = len(label_encoder.classes_)
print(f"Found {num_classes} unique classes.")

# Create mappings for later use
id_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
label_to_id = {label: i for i, label in enumerate(label_encoder.classes_)}

# STEP 4: SPLIT DATA

print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['combined_text'], # Use the new combined field
    df['queue_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=df['queue_encoded']
)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# STEP 5: TOKENIZATION WITH A TRANSFORMER-SPECIFIC TOKENIZER

print("\nTokenizing text for DistilBERT...")
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 170 

# Tokenize the training and test sets
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=MAX_LEN)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=MAX_LEN)

# STEP 6: CREATE TENSORFLOW DATASETS

print("\nCreating TensorFlow datasets for efficient training...")
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train.tolist()
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test.tolist()
))

# STEP 7: LOAD AND COMPILE THE TRANSFORMER MODEL

print(f"\nLoading pre-trained model: {MODEL_NAME}...")
model = TFDistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    id2label=id_to_label,
    label2id=label_to_id,
    from_pt=True # Explicitly load from a PyTorch checkpoint
)

# STEP 8: CREATE OPTIMIZER WITH LEARNING RATE SCHEDULER

print("\nCreating optimizer with a learning rate scheduler...")
EPOCHS = 8 
BATCH_SIZE = 16 

# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied by the total number of epochs.
num_train_steps = (len(X_train) // BATCH_SIZE) * EPOCHS
num_warmup_steps = int(0.1 * num_train_steps) # 10% of steps for warmup

# Create a learning rate scheduler with warmup and decay
optimizer, _ = create_optimizer(
    init_lr=5e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps
)

# Compile the model with the new optimizer
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

# STEP 9: FINE-TUNE THE MODEL

print("\nFine-tuning the model...")

# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

history = model.fit(
    train_dataset.shuffle(1000).batch(BATCH_SIZE),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=test_dataset.batch(BATCH_SIZE),
    class_weight=class_weights_dict
)
print("\nModel fine-tuning finished.")

# STEP 10: SAVE THE FINE-TUNED MODEL AND TOKENIZER

print("\nSaving the fine-tuned model and tokenizer for future use...")
save_directory = "./fine_tuned_ticket_classifier"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer have been saved to the directory: '{save_directory}'")

# STEP 11: EVALUATE THE MODEL

print("\nEvaluating the fine-tuned model...")
# Predict on the test set
logits = model.predict(test_dataset.batch(BATCH_SIZE)).logits
y_pred_classes = np.argmax(logits, axis=1)
y_test_classes = y_test.to_numpy()

# Calculate and print metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Transformer Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# STEP 11: GEMINI API INTEGRATION (DEMONSTRATION)

print("\nSetting up Gemini API for response generation...")
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
except (ImportError, KeyError):
    print("Could not find GEMINI_API_KEY in Colab secrets. Please paste your API key here.")
    try:
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        if not GEMINI_API_KEY:
             GEMINI_API_KEY = input("Paste your Gemini API key and press Enter: ")
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Could not configure Gemini API: {e}")
        GEMINI_API_KEY = None

def generate_customer_reply(ticket_text, predicted_queue):
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Cannot generate response."
    model_gen = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
    prompt = f"""
    You are a helpful and empathetic customer support assistant.
    A customer has submitted a support ticket which our system has classified into the "{predicted_queue}" queue.
    The customer's message is: "{ticket_text}"
    Your task is to draft a polite, generic, and reassuring initial response to the customer.
    The response should:
    1. Acknowledge their issue without promising a specific solution yet.
    2. Confirm that their ticket has been received and routed to the correct team.
    3. Reassure them that someone will get back to them soon.
    4. Maintain a professional and friendly tone.
    Generate the response now.
    """
    try:
        response = model_gen.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

print("\n--- Demonstrating Full Pipeline: Classify and Reply ---")
# Select a random sample from the original test data
sample_text = X_test.sample(1).iloc[0]

# Tokenize the single sample
predict_input = tokenizer.encode(
    sample_text,
    truncation=True,
    padding=True,
    return_tensors="tf"
)

# Predict
output = model(predict_input)[0]
prediction_value = tf.argmax(output, axis=1).numpy()[0]
predicted_queue = label_encoder.inverse_transform([prediction_value])[0]

# Generate reply
# We need to find the original ticket text for the sample to pass to Gemini
original_subject = df[df['combined_text'] == sample_text]['subject'].values[0]
original_body = df[df['combined_text'] == sample_text]['body'].values[0]
gemini_input_text = f"Subject: {original_subject}\n\nBody: {original_body}"

generated_reply = generate_customer_reply(gemini_input_text, predicted_queue)

print(f"\nOriginal Ticket Text (Combined for Model):\n\"{sample_text}\"")
print("-" * 30)
print(f"Predicted Queue: {predicted_queue}")
print("-" * 30)
print(f"Generated Reply (from Gemini):\n{generated_reply}")
print("-" * 30)
print("\nProject execution finished.")
