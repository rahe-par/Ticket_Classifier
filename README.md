Automatic Ticket Classification and Generative AI Reply System

# Problem Statement
Organizations today are inundated with thousands of customer support tickets daily across various channels. The initial step of triaging—reading each ticket and routing it to the correct department (e.g., Billing, Technical Support, Sales)—is a significant bottleneck. Misclassification can lead to delayed resolutions, frustrated customers, and increased operational costs. The goal is to automate this classification process to improve efficiency, speed, and customer satisfaction.

# Project Objectives
The primary objectives of this project were to:

Develop a machine learning model capable of automatically classifying customer support tickets based on their text content.
Address the significant class imbalance present in the real-world dataset.
Achieve the highest possible accuracy by employing state-of-the-art Natural Language Processing (NLP) techniques.
Integrate the classification model with a generative AI to automate the initial customer acknowledgment and reply.
Create a complete, end-to-end prototype pipeline that can receive a ticket, classify it, and generate a response.

# Dataset Explanation
Source: Hugging Face Hub
Dataset Name: Tobi-Bueck/customer-support-tickets
Format: The dataset consists of structured entries, with each entry representing a single customer support ticket.

# Key Features
The primary features used for this project were:

subject: The subject line of the ticket, providing a concise summary.
body: The full text of the customer's message.
queue: The target label, representing the correct department or queue for the ticket.

# Gemini API Integration for Automated Replies
The final step involved creating a function to generate customer replies.

Prediction: The trained model predicts the queue for a given ticket.
Prompt Engineering: The model's prediction and the original ticket text are fed into a carefully crafted prompt for the Gemini language model (gemini-2.5-flash-preview-05-20).
Generation: The prompt instructs Gemini to act as a support assistant and generate a polite, empathetic, and reassuring initial response, confirming that the ticket has been routed to the correct team.

# Results and Evaluation

The fine-tuned XLM-RoBERTa base model achieved a final Test Accuracy of 0.6891%.

This project is for educational purposes as part of the GUVI Artificial Intelligence and Machine Learning program.
