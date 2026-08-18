# Bilingual PDF RAG Bot

> **Project Status:** Active
> **Tech Stack:** LangChain, Streamlit, Google Gemini, FAISS, Python
> **Target:** Arabic & English Document Question Answering

## Overview

Built a bilingual Retrieval-Augmented Generation (RAG) web application for querying multi-page PDF documents, including legal contracts, terms and conditions, and financial reports, using natural-language questions in Arabic or English.

## Key Features

* **Bilingual Semantic Retrieval:** Processes Arabic and English documents using multilingual Sentence-Transformers embeddings.
* **Smart Vector Caching:** Automatically chunks documents, generates local embeddings, and caches FAISS vector stores to avoid repeated processing of previously uploaded documents.
* **Context-Grounded QA:** Uses LangChain Expression Language (LCEL) to retrieve the top 3 relevant chunks and provide them as context to the LLM, reducing unsupported answers.
* **Document-Aware Responses:** Restricts generated answers to information retrieved from the uploaded documents.
* **Streamlit Interface:** Provides an interactive chat interface for uploading PDFs and asking questions in real time.

## Architecture

```text
PDF Document
    ↓
Text Extraction
    ↓
Recursive Character Text Splitting
    ↓
Multilingual Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Search — Top K=3
    ↓
Retrieved Context
    ↓
Prompt + Context
    ↓
Google Gemini
    ↓
Grounded Answer
    ↓
Streamlit Chat Interface
```

## Tech Stack

| Category       | Technology                              |
| -------------- | --------------------------------------- |
| Frontend / UI  | Streamlit                               |
| RAG Framework  | LangChain / LCEL                        |
| PDF Processing | PyPDF2                                  |
| LLM            | Google Gemini                           |
| Embeddings     | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Store   | FAISS                                   |
| Language       | Python                                  |

## CV Highlights

* Built a **bilingual Arabic/English RAG application** using LangChain, Streamlit, Gemini, FAISS, and Sentence-Transformers for document question answering.
* Implemented **multilingual semantic retrieval** with locally generated embeddings and FAISS vector search.
* Developed **persistent vector-store caching** to prevent repeated document processing and embedding generation for previously uploaded PDFs.
* Designed **LCEL-based retrieval and generation chains** that provide the top 3 relevant document chunks as context to the LLM.
* Implemented **context-grounded prompting** to restrict responses to information available in the uploaded documents.
* Used `RecursiveCharacterTextSplitter` with a **1000-character chunk size and 200-character overlap** to balance retrieval accuracy and LLM context size.

## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/bilingual-pdf-rag-bot.git
cd bilingual-pdf-rag-bot

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Run the application:

```bash
streamlit run app.py
```
