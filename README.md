# Clinical NLP Portfolio

A portfolio project demonstrating expertise in Natural Language Processing (NLP), clinical data analysis, ETL, predictive modeling, and client-facing reporting for healthcare research.

## Project Overview
This repository showcases:

## Workflow Diagram

Below is a visual summary of the end-to-end workflow using Mermaid:

```mermaid
flowchart TD
	A[Start: Synthetic Clinical Data] --> B[ETL: Load & Clean Data]
	B --> C[NLP: Rule-based Extraction]
	B --> D[NLP: spaCy NER]
	C --> E[Feature Engineering]
	D --> E
	E --> F[Predictive Modeling - Logistic Regression]
	F --> G[Model Evaluation & Visualization]
	G --> H[Interpretation & Reporting]
	F --> I{Disease Examples}
	I --> J[Diabetes]
	I --> K[Hypertension]
	I --> L[Asthma]
	I --> M[Heart failure]
	I --> N[COPD]
	I --> O[Depression]
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
