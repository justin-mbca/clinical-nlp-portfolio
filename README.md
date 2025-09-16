# Clinical & GenAI NLP Portfolio

A comprehensive portfolio project demonstrating advanced expertise in Clinical Natural Language Processing (NLP), Generative AI, and data science. This repository showcases the complete lifecycle of a healthcare AI project, from ETL and de-identification of synthetic clinical data to predictive modeling, agentic architecture design, and client-ready reporting.

*   **Advanced NLP & LLMs:** Fine-tuning and deploying domain-specific models like Bio_ClinicalBERT for clinical entity recognition.
*   **Generative AI:** Implementing Retrieval-Augmented Generation (RAG), vector databases, and prompt engineering for clinical Q&A.
*   **Agentic Systems:** Designing multi-agent architectures with standardized protocols (A2A, MCP) for automated workflows.
*   **End-to-End ML Pipeline:** Expertise in data ETL, feature engineering, predictive modeling (e.g., logistic regression for disease risk), and evaluation.
*   **Healthcare Compliance:** Built-in de-identification and adherence to healthcare data standards (FHIR, ICD).

## 🧠 Conceptual Overview: A Guided Tour

This project integrates several advanced concepts. This section breaks them down to provide a clear mental model of how everything fits together.

#### The Core Idea (The "What")
This project simulates an **AI system for healthcare** that can read clinical notes, understand their meaning, predict patient risks, and answer medical questions through collaborative AI agents.

#### The Five Key Technologies (The "How")
1.  **🤖 NLP & LLMs (Natural Language Processing & Large Language Models):**
    *   **Think of it as:** Teaching a computer to read medical jargon.
    *   **In this project:** We use **Bio_ClinicalBERT**, a powerful model pre-trained on medical text, to act as a super-intern that expertly identifies and highlights medical terms like diseases and symptoms in clinical notes. This is called Named Entity Recognition (NER).

2.  **📚 RAG (Retrieval-Augmented Generation):**
    *   **Think of it as:** A "smart search" for AI. Instead of guessing, the AI first *looks up* facts and then *formulates* an answer.
    *   **In this project:** When asked a complex clinical question, the system first retrieves the most relevant information from a trusted source (like medical guidelines stored in a database) and then uses an LLM to generate a accurate, sourced answer. This is crucial for safety in medicine.

3.  **👥 Multi-Agent Systems:**
    *   **Think of it as:** An automated workforce of specialized AI assistants that talk to each other.
    *   **In this project:** Different agents have different jobs (e.g., `ClaimsAgent`, `EligibilityAgent`). They collaborate by passing messages using **A2A (Agent-to-Agent)** protocols and remember their shared context using **MCP (Model Context Protocol)**, much like different hospital departments sharing patient files.

4.  **🧠 VectorDB (Vector Database):**
    *   **Think of it as:** The system's long-term, semantic memory.
    *   **In this project:** It stores medical knowledge and past cases. Its superpower is **semantic search**—it can find information based on conceptual meaning, not just keywords. This powers the RAG system and provides memory for the agents.

5.  **📊 Predictive Modeling:**
    *   **Think of it as:** Finding patterns in data to forecast future outcomes.
    *   **In this project:** After extracting medical concepts from text, we use statistical models (like Logistic Regression) to identify patterns that predict a patient's risk of developing a disease like diabetes or hypertension.

#### The Project Narrative (The "Story")
The workflow follows a logical, end-to-end pipeline:
**Data -> Clean -> Protect -> Find -> Build -> Predict -> Report**
1.  **Data:** Begin with **synthetic** (artificial) clinical data for safety.
2.  **Clean & Protect:** Process it through **ETL**, then **de-identify** it to remove any simulated personal information.
3.  **Find:** Use **NER** (with our expert "Bio_ClinicalBERT intern") to find crucial medical information in the text.
4.  **Build:** Use those findings to build patient **cohorts** (groups) and create features for modeling (**feature engineering**).
5.  **Predict:** Train a **predictive model** to assess disease risk.
6.  **Report:** **Evaluate** the model's performance, create visualizations, and generate a **client-ready analysis**.

---

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
│   ├── advanced_genai_features_demo.ipynb  # Advanced GenAI/NLP demo notebook
│   ├── clinical_nlp_agents_demo.ipynb      # Modular agentic architecture demo
│   ├── clinical_nlp_demo.ipynb             # Classic clinical NLP workflow demo
│   └── clinical_chatbot_demo.py            # Clinical Q&A chatbot demo (Hugging Face QA)
## 🤖 Clinical Q&A Chatbot Demo

The notebook/script `notebooks/clinical_chatbot_demo.py` demonstrates a simple clinical question-answering chatbot using Hugging Face's question-answering pipeline (e.g., `deepset/roberta-base-squad2`).

**Features:**
- Answers clinical questions based on a provided medical context (e.g., symptoms, diseases, treatments)
- Uses open-source models for local, privacy-preserving Q&A
- Easily extensible with custom medical FAQs or context documents

**Example:**
```
Ask a clinical question: What are the symptoms of diabetes?
Agent: increased thirst, frequent urination, fatigue, and blurred vision
```
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


## 📓 Notebooks Overview

### 1. advanced_genai_features_demo.ipynb
Recruiter-ready deep dive into advanced GenAI and clinical NLP workflows:
* Entity extraction & classification with Bio_ClinicalBERT
* Retrieval-Augmented Generation (RAG) pipeline
* VectorDB integration (Annoy, FAISS)
* Prompt engineering & finetuning
* Bias detection & safety
* Cloud integration (AWS S3)
* PEFT/SFT finetuning (LoRA, Hugging Face PEFT)

```mermaid
flowchart TD
    A(Start: Clinical Text) --> B(Entity Extraction: Bio_ClinicalBERT)
    A --> C(Retrieval-Augmented Generation: RAG)
    A --> D(Vector DB Search: Annoy/FAISS)
    B --> E(Prompt Engineering)
    C --> E
    D --> E
    E --> F(Bias Detection & Safety)
    F --> G(Cloud Integration: AWS S3)
    G --> H(PEFT/SFT Finetuning)
    H --> I(Model Deployment)
```

### 2. clinical_nlp_agents_demo.ipynb
Modular agentic architecture demo:
* Each agent (Claims, Eligibility, De-identification, Cohort Phenotyping, Coding, Retrieval, Summary, Risk Prediction, Care Coordination) is implemented in a dedicated section
* Demonstrates agent-to-agent protocols (A2A), Model Context Protocol (MCP), RAG, FHIR/ICD standards, and vector DB integration
* Includes markdown explanations, code, and orchestration for real-world clinical operations

### 3. clinical_nlp_demo.ipynb
Classic clinical NLP workflow demo:
* End-to-end workflow for synthetic clinical data
* ETL: Load and clean data
* NLP: Extract disease phenotypes and symptoms
* Analysis: Visualize and summarize results
* Predictive modeling: Disease risk classification (diabetes, hypertension, asthma, etc.)
* Feature engineering, model evaluation, and interpretation

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