# RAG Benchmark & Evaluation Platform

A modular Retrieval-Augmented Generation (RAG) benchmark and evaluation platform for comparing retrieval strategies and measuring retrieval quality using a curated evaluation dataset.

The platform currently supports Dense Retrieval, BM25, and Hybrid Retrieval using Reciprocal Rank Fusion (RRF), along with automated evaluation and per-query error analysis.

---

## Overview

This project was built to study and evaluate how different retrieval strategies perform within a RAG system.

Instead of treating a RAG system as a single pipeline, the project separates document processing, retrieval, evaluation, and generation into independent components.

The current system supports:

- PDF document ingestion
- Text normalization
- Fixed-size chunking with overlap
- Sentence Transformer embeddings
- FAISS vector search
- Dense retrieval
- BM25 sparse retrieval
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Curated retrieval evaluation dataset
- Recall@K
- Precision@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Per-query evaluation
- Retrieval error analysis
- Local LLM-based answer generation using Ollama

---

## Problem Statement

RAG systems depend heavily on the quality of their retrieval component.

A retrieved context can contain relevant information, irrelevant information, or miss important information entirely. Therefore, simply building a RAG chatbot does not provide a reliable way to understand whether the retrieval system is performing well.

This project focuses on building a benchmark that can:

1. Compare different retrieval strategies.
2. Measure retrieval quality using standard information-retrieval metrics.
3. Identify queries where relevant chunks were missed.
4. Identify false-positive retrieved chunks.
5. Provide a reproducible foundation for experimenting with future retrieval techniques.

---

## Architecture

                         PDF Documents
                              |
                              v
                         PDF Loader
                              |
                              v
                       Text Normalization
                              |
                              v
                           Chunking
                              |
                              v
                         Chunk Objects
                              |
                    +---------+---------+
                    |                   |
                    v                   v
             Dense Retrieval         BM25
                    |                   |
             Sentence Transformer      |
                    |                   |
                  FAISS                 |
                    |                   |
                    +---------+---------+
                              |
                              v
                    Hybrid Retrieval
                         (RRF)
                              |
                              v
                     Retrieved Chunks
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                 Evaluation          RAG Pipeline
                    |                   |
                    |             Prompt Builder
                    |                   |
                    |                Ollama
                    |                   |
                    |                   v
                    |              LLM Answer
                    |
                    v
              Retrieval Metrics
                    |
                    v
              Error Analysis