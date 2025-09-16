# Clinical NLP Portfolio

A portfolio project demonstrating expertise in Natural Language Processing (NLP), clinical data analysis, ETL, predictive modeling, and client-facing reporting for healthcare research.

## Project Overview
This repository showcases:
- End-to-end clinical NLP workflow on synthetic EHR data
- Advanced Named Entity Recognition (NER) using transformer-based models (Bio_ClinicalBERT)
- Rule-based cohort phenotyping for disease risk
- De-identification of clinical notes for privacy
- Feature engineering and predictive modeling
- Interactive Jupyter notebook workflow for demonstration

## Agentic Architectures & Protocols Notebook

The `clinical_nlp_agents_demo.ipynb` notebook demonstrates advanced agentic architectures and protocols for healthcare AI, including:
- Agent-to-Agent (A2A) protocols for autonomous agent collaboration
- Model Context Protocol (MCP) for persistent, context-aware agent memory
- Retrieval-Augmented Generation (RAG) for dynamic knowledge retrieval and LLM-based answer generation
- Parsing and handling healthcare data standards (FHIR, ICD)
- VectorDB integration for agent memory and retrieval

This notebook is designed to showcase expertise in building interoperable, context-aware, and self-improving agentic systems for healthcare operations.

### Agentic Architecture Overview (Mermaid)

```mermaid
flowchart TD
	A[ClaimsAgent] -- A2A Protocol --> B[EligibilityAgent]
	B -- A2A Protocol --> C[ProviderMatchAgent]
	A -- MCP Context --> D[Model Context Memory]
	B -- MCP Context --> D
	C -- MCP Context --> D
	A -- RAG Query --> E[Document Retriever]
	E -- Retrieved Docs --> F[LLM Generator]
	F -- Answer --> A
	D -- VectorDB --> G[Vector Memory]
	G -- Memory Retrieval --> A
	G -- Memory Retrieval --> B
```

## Workflow Diagram

Below is a visual summary of the end-to-end workflow using Mermaid:

```mermaid
flowchart TD
	A[Start: Synthetic Clinical Data] --> B[ETL: Load & Clean Data]
	B --> C[De-identification]
	C --> D[Transformer-based NER: Bio_ClinicalBERT]
	C --> E[spaCy NER]
	C --> F[Rule-based Extraction]
	D --> G[Feature Engineering]
	E --> G
	F --> G
	G --> H[Rule-based Cohort Builder]
	G --> I[Predictive Modeling - Logistic Regression]
	I --> J[Model Evaluation & Visualization]
	J --> K[Interpretation & Reporting]
	I --> L{Disease Examples}
	L --> M[Diabetes]
	L --> N[Hypertension]
	L --> O[Asthma]
	L --> P[Heart failure]
	L --> Q[COPD]
	L --> R[Depression]
```

## Structure
- `data/` – Sample clinical notes and EHR data (de-identified)
- `notebooks/` – Jupyter notebooks for exploration, modeling, and reporting
- `src/` – Python modules for ETL, NLP, modeling, and utilities
- `reports/` – Generated reports and client deliverables

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Explore notebooks in `notebooks/`
4. Run ETL and modeling scripts in `src/`

## Example Use Cases
- Extract disease phenotypes from clinical notes
- Predict patient risk for conditions
- Generate client-ready reports

## Advanced NLP Features

## Large Language Model (LLM) Usage: Bio_ClinicalBERT

This project leverages Bio_ClinicalBERT, a domain-specific large language model (LLM) based on BERT and pre-trained on clinical and biomedical text. Bio_ClinicalBERT is used for advanced Named Entity Recognition (NER) to extract disease and symptom entities from clinical notes with high accuracy.

**How Bio_ClinicalBERT is used:**
- Loaded via Hugging Face Transformers in the notebook and code modules
- Used for direct inference (entity extraction) on clinical notes
- Optionally fine-tuned with custom labeled data for specialized NER tasks
- Compared with spaCy and rule-based extraction for benchmarking

**Why Bio_ClinicalBERT?**
- Recognizes medical terminology and entities better than general-purpose models
- Widely adopted in healthcare AI research and production

See the notebook for code examples of both inference and (optional) fine-tuning with Bio_ClinicalBERT.

## Advanced NLP Features
- Fine-tune and evaluate transformer-based NER models on clinical notes (`run_advanced_ner`)
- Use rule-based logic for cohort selection (e.g., T2D phenotyping, `assess_t2d_cohort`)
- De-identify notes before NLP and modeling (`deidentify_text`)
- Compare transformer-based NER with spaCy and rule-based extraction
- Visualize and interpret model outputs in the notebook

## Compliance
This project simulates privacy and IRB compliance for demonstration purposes. No real patient data is used.

## Further Development

## Expanding Disease Risk Prediction

This pipeline can be easily extended to predict risk for other diseases. Currently, diabetes, hypertension, and asthma are included. To add more:

- Update the target variable to the disease of interest (e.g., heart failure, COPD, depression)
- Use the same feature engineering and modeling steps
- For multi-class prediction, use multiclass classifiers (e.g., multinomial logistic regression, random forest)

**Examples of diseases to expand:**
- Heart failure
- COPD
- Depression
- Any disease present in the dataset

See the notebook for code examples and guidance.

## Contact
For more information, reach out to Justin.
