# Clinical Q&A Chatbot Demo with LangChain and Hugging Face

from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from transformers import pipeline

# Load a local Hugging Face model (choose a small model for local use, e.g., distilgpt2)






# Use a valid question-answering model from Hugging Face
model_name = "deepset/roberta-base-squad2"  # General QA model, works for medical context

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model_name)

print("Clinical Q&A Chatbot. Type 'exit' to quit.")
# Example medical context (expand or load from a medical FAQ/document)
context = "Diabetes is a chronic condition characterized by high blood sugar levels. Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision. Hypertension is high blood pressure, often with no symptoms but can lead to heart disease. Asthma causes wheezing and shortness of breath."

while True:
    user_input = input("Ask a clinical question: ")
    if user_input.lower() == "exit":
        break
    result = qa_pipeline(question=user_input, context=context)
    print("Agent:", result['answer'])
