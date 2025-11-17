# Biomedical RAG Agent – Vision Document

## 1. Introduction

### 1.1 Purpose
The purpose of this document is to describe the vision of the Biomedical Retrieval-Augmented Generation Agent project.  
It defines the project goals, target users, major features, success criteria, and constraints.  
The vision establishes a common understanding among stakeholders and serves as a foundation for all future requirements and design activities.

### 1.2 Scope
The Biomedical RAG Agent aims to simplify and enhance access to biomedical research literature by combining semantic search with generative AI.  
The system will retrieve the most relevant, source-grounded information from a curated biomedical corpus and generate concise, explainable summaries with citations.

**Key components:**
- Biomedical document ingestion and metadata extraction  
- Semantic search using a vector database (Qdrant)  
- Retrieval-Augmented Generation (RAG) model 
- Evaluation core of 4,000 labeled documents for performance benchmarking  
- REST API and simple UI for researcher interaction  

### 1.3 Definitions, Acronyms, and Abbreviations
- **RAG** – Retrieval-Augmented Generation  
- **LLM** – Large Language Model  
- **Qdrant** – Vector databases for similarity search  
- **Precision@k / Recall@k** – Information retrieval performance metrics  
- **Groundedness** – Degree to which an answer is supported by retrieved evidence  

### 1.4 References
- IBM Engineering Lifecycle Management: *Vision Document Template*  
- OpenAI API Documentation  

&nbsp;

## 2. Positioning

### 2.1 Business Opportunity
Biomedical research output is vast and growing, with millions of publications and complex terminologies.  
Traditional keyword-based systems (like BM25) struggle to capture semantic meaning, synonyms, and context.  
Researchers need an intelligent system that can understand biomedical language, locate relevant evidence, and generate accurate, explainable answers.

### 2.2 Problem Statement
Biomedical researchers face information overload, low recall from traditional search tools, and limited trust in AI-generated answers.  
The Biomedical RAG Agent addresses these issues by providing:
- High-recall semantic retrieval  
- Transparent, citation-based responses  
- Evaluation-driven accuracy metrics  

### 2.3 Product Position Statement
For biomedical researchers, clinicians, and PhD students who need reliable, evidence-based insights from scientific literature, Biomedical RAG Agent is an AI-powered search and summarization platform that delivers accurate, grounded, and explainable answers with references. Unlike standard search engines, it combines vector-based retrieval and generative reasoning to ensure trust and transparency.

&nbsp;

## 3. Stakeholder and User Descriptions

### 3.1 Stakeholders
| Stakeholder | Role |
|--------------|------|
| Prof. Dr. Markus Goldstein | Project Supervisor (THU) |
| Prof. Dr. Christian Beltinger | Customer (UKU) |

### 3.2 User Environment
Users interact with the system through a web interface or API. They can:
- Submit natural language biomedical queries  
- Receive answers with cited sources  
- Inspect retrieved passages and trust levels  

&nbsp;

## 4. Product Overview

### 4.1 Product Perspective
The Biomedical RAG Agent is a standalone research tool built using **Python**, **FastAPI**, and **LangGraph**.  
It interfaces with vector databases (Qdrant) for retrieval and OpenAI models (GPT-5mini/5) for generation.

### 4.2 Product Functions
1. **Document Management:** Ingest and store biomedical documents with metadata.  
2. **Semantic Chunking:** Divide documents into overlapping sections for embedding.  
3. **Vector Search:** Retrieve top-k relevant passages semantically.  
4. **RAG Answering:** Generate grounded answers with citations.   
5. **Evaluation Core:** Benchmark performance on 4,000 labeled samples.  
6. **API & UI:** Allow external queries and visualization of results.  

### 4.3 Operating Environment
- **Language:** Python 3.11  
- **Frameworks:** FastAPI, LangGraph  
- **Vector Store:** Qdrant  
- **LLM:** OpenAI GPT-5mini/5
- **Deployment:** Local or cloud environment  

### 4.4 Design and Implementation Constraints
- API budget: ≤ €500 total usage  
- Time frame: 7 weeks  
- Data sources: Open-access biomedical literature only  

### 4.5 Assumptions and Dependencies
- Reliable API access (OpenAI)  
- Sufficient local storage for vector embeddings (120k documents)  
- Team collaboration via GitHub  

&nbsp;

## 5. Product Features

| Feature | Description | Priority |
|----------|--------------|-----------|
| Document ingestion | Load and parse biomedical texts with metadata | High |
| Chunking and embedding | Split text and create embeddings for semantic search | High |
| Vector retrieval | Query embeddings for top-k relevant results | High |
| RAG answer generation | Generate concise answers grounded in retrieved texts | High |
| Writer–Critic–Judge loop | Validate and refine generated answers | Medium |
| Evaluation module | Measure recall, precision, groundedness | Medium |
| FastAPI interface | REST endpoints for retrieval and answer generation | High |
| Web UI | Visualize questions, answers, and evidence | Low |
| Cost tracking | Monitor API token usage and cost | Low |

&nbsp;


## 6. Risks and Countermeasures

| Risk | Impact | Countermeasure |
|------|---------|-------------|
| Hallucinated answers | High | Enforce source-based generation |
| API cost overrun | Medium | Monitor token use; use GPT-5mini for testing |
| Low retrieval recall | Medium | Experiment with hybrid retrieval or reranking |
| Schedule slippage | Medium | Weekly sprint reviews and checkpoints |
| Data access issues | Low | Use open-access biomedical datasets |

&nbsp;

## 7. Appendices
- **Team Members:** Erind Bylyku, Firas Bazerbashi, Luke Weiss  
- **Version Control:** https://github.com/Lutherpick/Biomedical-RAG-Agent.git

&nbsp;

## 8. Team Organization

During the week every person works individually on their tasks.
The team meets twice a week to coordinate development and ensure consistent progress.  
During these meetings, members discuss completed tasks, review ongoing work, and plan the next steps for each milestone.
