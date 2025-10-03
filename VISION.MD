# Vision

The Biomedical RAG Agent project aims to make biomedical research easier to find and understand.  
It combines a **vector database (Qdrant)** and a **Retrieval-Augmented Generation (RAG) agent** to give researchers accurate, source-based answers.

## Why?
- Biomedical texts are huge and complex (120k+ documents).
- Keyword searches (like BM25) are not enough, since they miss synonyms and meanings.
- Researchers need trustworthy results with full source evidence.

## Goals
1. Store biomedical documents (with metadata).
2. Split long documents into chunks for better search.
3. Label a smaller “Evaluation Core” of 1,000 docs to test accuracy.
4. Let users query in natural language and get top-5 relevant results.
5. Evaluate the system with recall, precision, and groundedness.
