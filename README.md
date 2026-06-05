# Healthcare Policy RAG Navigator

A Retrieval-Augmented Generation (RAG) system for querying healthcare policy and Medicare documentation using hybrid retrieval and reranking techniques.

## Features

* Hybrid Retrieval using FAISS and BM25
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Reranking
* OpenAI Embeddings
* Retrieval Evaluation Pipeline
* FHIR-based Healthcare Data Integration

## Tech Stack

* Python
* FAISS
* BM25
* Cross Encoders
* OpenAI Embeddings
* AWS
* FHIR

## Architecture

Query → Hybrid Retrieval (FAISS + BM25) → RRF → Cross Encoder Reranker → Context Selection → LLM Response

## Key Learnings

* Retrieval-Augmented Generation (RAG)
* Embeddings and Semantic Search
* Hybrid Search Architectures
* Retrieval Evaluation
* Healthcare Data Standards (FHIR)

## Future Improvements

* Agentic workflows using LangGraph
* Production deployment
* Advanced evaluation metrics
* Real-time document ingestion
