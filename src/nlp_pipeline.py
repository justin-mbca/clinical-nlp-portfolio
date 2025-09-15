"""
Module: nlp_pipeline.py
Extracts disease phenotypes and symptoms from clinical notes using basic NLP.
"""
import pandas as pd
import re

def extract_phenotypes(note, diseases):
    found = [d for d in diseases if d in note.lower()]
    return found

def extract_symptoms(note, symptoms):
    found = [s for s in symptoms if s in note.lower()]
    return found

def nlp_process(df, diseases, symptoms):
    df["extracted_diseases"] = df["clinical_note"].apply(lambda x: extract_phenotypes(x, diseases))
    df["extracted_symptoms"] = df["clinical_note"].apply(lambda x: extract_symptoms(x, symptoms))
    return df

if __name__ == "__main__":
    DISEASES = ["diabetes", "hypertension", "asthma", "copd", "heart failure", "depression"]
    SYMPTOMS = ["fatigue", "cough", "chest pain", "shortness of breath", "headache", "nausea"]
    df = pd.read_csv("../data/synthetic_clinical_data.csv")
    df = nlp_process(df, DISEASES, SYMPTOMS)
    print(df[["clinical_note", "extracted_diseases", "extracted_symptoms"]].head())
