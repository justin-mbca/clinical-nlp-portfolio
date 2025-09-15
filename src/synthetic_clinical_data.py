"""
Module: synthetic_clinical_data.py
Generates synthetic clinical notes and tabular EHR data for NLP and modeling workflows.
"""
import random
import pandas as pd
from faker import Faker

fake = Faker()

DISEASES = ["diabetes", "hypertension", "asthma", "COPD", "heart failure", "depression"]
SYMPTOMS = ["fatigue", "cough", "chest pain", "shortness of breath", "headache", "nausea"]


def generate_patient_record(patient_id):
    age = random.randint(18, 90)
    gender = random.choice(["M", "F"])
    disease = random.choice(DISEASES)
    symptoms = random.sample(SYMPTOMS, k=random.randint(1, 3))
    note = f"Patient presents with {', '.join(symptoms)}. History of {disease}."
    return {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "disease": disease,
        "symptoms": ", ".join(symptoms),
        "clinical_note": note,
    }


def generate_synthetic_data(num_patients=100):
    records = [generate_patient_record(i + 1) for i in range(num_patients)]
    df = pd.DataFrame(records)
    return df


def save_synthetic_data(path="data/synthetic_clinical_data.csv", num_patients=100):
    df = generate_synthetic_data(num_patients)
    df.to_csv(path, index=False)
    print(f"Synthetic data saved to {path}")


if __name__ == "__main__":
    save_synthetic_data()
