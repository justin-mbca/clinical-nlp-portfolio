# Clinical & GenAI NLP Portfolio

A comprehensive portfolio project demonstrating advanced expertise in Clinical Natural Language Processing (NLP), Generative AI, and data science. This repository showcases the complete lifecycle of a healthcare AI project, from ETL and de-identification of synthetic clinical data to predictive modeling, agentic architecture design, and client-ready reporting.

## 🚀 Value Proposition

This project is designed to simulate a real-world healthcare AI research environment, highlighting key skills:
*   **Advanced NLP & LLMs:** Fine-tuning and deploying domain-specific models like Bio_ClinicalBERT for clinical entity recognition.
*   **Generative AI:** Implementing Retrieval-Augmented Generation (RAG), vector databases, and prompt engineering for clinical Q&A.
*   **Agentic Systems:** Designing multi-agent architectures with standardized protocols (A2A, MCP) for automated workflows.
*   **End-to-End ML Pipeline:** Expertise in data ETL, feature engineering, predictive modeling (e.g., logistic regression for disease risk), and evaluation.
*   **Healthcare Compliance:** Built-in de-identification and adherence to healthcare data standards (FHIR, ICD).

## ✨ Features

*   **Transformer-Based NER:** Extraction of medical concepts using Bio_ClinicalBERT from Hugging Face.
*   **Multi-Agent Architecture:** Demonstrates agent-to-agent (A2A) communication and context management with Model Context Protocol (MCP).
*   **Retrieval-Augmented Generation (RAG):** A working RAG pipeline for answering complex clinical questions.
*   **Vector Database Integration:** Semantic search and efficient memory retrieval using Annoy and FAISS/Chroma.
*   **Predictive Modeling:** Logistic regression models for predicting patient disease risk (e.g., Diabetes, Hypertension).
*   **Healthcare Interoperability:** Parsers and utilities for FHIR and ICD standards.
*   **Cloud & MLOps Ready:** Examples and patterns for deploying models to AWS and implementing MLOps practices.

## 📁 Project Structure

```
.
├── data/               # Sample synthetic clinical notes and EHR data (de-identified)
├── notebooks/          # Jupyter notebooks for exploration, modeling, and reporting
│   └── advanced_genai_features_demo.ipynb  # Main demo notebook
├── src/                # Python source modules
│   ├── etl.py          # Data loading and cleaning utilities
│   ├── nlp.py          # NLP functions (NER, de-identification)
│   ├── modeling.py     # Predictive modeling routines
│   └── utils.py        # Helper functions
├── reports/            # Generated client deliverables and reports
└── requirements.txt    # Python dependencies
```

## 🏗️ Agentic Architectures & Protocols

The project implements a forward-looking multi-agent architecture for clinical decision support.

### Architectural Overview
```mermaid
graph TD
    A2A[Agent-to-Agent Protocols] --> MCP[Model Context Protocol]
    A2A --> RAG[Retrieval-Augmented Generation]
    A2A --> FHIR[Healthcare Data Standards]
    A2A --> VectorDB[VectorDB Integration]
    MCP --> RAG
    MCP --> VectorDB
    RAG --> FHIR
    RAG --> VectorDB
    FHIR --> VectorDB
    subgraph Agents
        A2A
        MCP
        RAG
        FHIR
        VectorDB
    end
```

### Example Agent Workflow
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

## 🔬 Technical Workflow

The end-to-end data science and NLP pipeline is visualized below:

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

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd clinical-nlp-portfolio
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (`venv` or `conda`).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Explore the project:**
    *   The main demonstration is in `notebooks/advanced_genai_features_demo.ipynb`.
    *   Run ETL and modeling scripts from the `src/` directory.

## 📓 Advanced GenAI Features Demo Notebook

The flagship notebook (`notebooks/advanced_genai_features_demo.ipynb`) is a recruiter-ready deep dive into modern clinical AI, showcasing:

**Section Overview:**
1.  **Entity Extraction & Classification:** Advanced NER with Hugging Face's Bio_ClinicalBERT.
2.  **RAG Pipeline:** A complete Retrieval-Augmented Generation system for clinical question answering.
3.  **VectorDB Integration:** Semantic search implementations with Annoy and FAISS.
4.  **Prompt Engineering & Finetuning:** Strategies for optimizing LLM interactions and conceptual fine-tuning workflows.
5.  **Bias Detection & Safety:** Implementing model guardrails and fairness checks.
6.  **Cloud Integration (AWS):** Code examples for model and data storage in S3.
7.  **PEFT/SFT Finetuning:** Setup for parameter-efficient fine-tuning using Hugging Face PEFT and LoRA.

The notebook workflow is modular, allowing sections to be understood and run independently.

```mermaid
flowchart TD
    A[Start: Clinical Text] --> B[Entity Extraction Bio_ClinicalBERT]
    A --> C[Retrieval-Augmented Generation RAG]
    A --> D[Vector DB Search Annoy/FAISS]
    B --> E[Prompt Engineering]
    C --> E
    D --> E
    E --> F[Bias Detection and Safety]
    F --> G[Cloud Integration AWS S3]
    G --> H[PEFT SFT Finetuning]
    H --> I[Model Deployment]
```

## 🔮 Further Development

The pipeline is designed for extension:
*   **New Disease Prediction:** Easily adapt the model to predict heart failure, COPD, depression, or any condition present in the data by modifying the target variable.
*   **New Agents:** Implement additional agents for tasks like prior authorization or clinical trial matching using the established A2A protocol.
*   **Real Data Integration:** The architecture supports plugging in real FHIR servers or EHR data streams.

## ⚠️ Compliance Note

**This project uses only synthetic, de-identified data.** It is designed to simulate privacy-aware development practices and IRB compliance for demonstration purposes. No real Protected Health Information (PHI) is used.

## 📧 Contact

For questions or more information, please reach out to **Justin**.

---