"""
Retrieval-Augmented Generation (RAG) Pipeline
Integrates document retrieval with LLM-based generation.
"""

from transformers import pipeline

class SimpleRAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer_query(self, query, documents):
        retrieved = self.retriever(query, documents)
        return self.generator(retrieved)
