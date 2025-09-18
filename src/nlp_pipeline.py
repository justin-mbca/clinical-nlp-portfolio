# Sample process_note function for notebook demo
def process_note(note):
    """Advanced demo: returns length, preview, spaCy NER, and clinical entities from Transformers."""
    result = {
        'length': len(note),
        'preview': note[:50]
    }
    # spaCy NER
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(note)
        result['spacy_entities'] = [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        result['spacy_entities'] = f"spaCy NER error: {e}"
    # Hugging Face Transformers clinical NER
    try:
        from transformers import pipeline
        ner_pipe = pipeline('ner', model='emilyalsentzer/Bio_ClinicalBERT', aggregation_strategy="simple")
        hf_entities = ner_pipe(note)
        result['clinical_entities'] = [
            {'entity_group': ent.get('entity_group', ''), 'word': ent['word'], 'score': ent['score']} for ent in hf_entities
        ]
    except Exception as e:
        result['clinical_entities'] = f"Transformers NER error: {e}"
    return result
"""
Module: nlp_pipeline.py
Extracts disease phenotypes and symptoms from clinical notes using basic NLP.
"""
"""
Advanced NLP: Adds spaCy NER and Hugging Face Transformers clinical entity extraction.
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
