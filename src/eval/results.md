
| Dataset                        | Chunking Method        | Context Precision | Context Recall | Answer Relevancy | Faithfulness |
|---------------------------------|------------------------|-------------------|----------------|------------------|--------------|
| **70 General QA Pairs**         | **CharacterTextSplitter**       | 0.21              | 0.22           | 0.40             | 0.36         |
|                                 | **RecursiveCharacterTextSplitter**| 0.30              | 0.31           | 0.45             | 0.45         |
|                                 | **Semantic Chunking**  | 0.18              | 0.20           | 0.35             | 0.32         |
| **100 Specific QA Pairs**       | **CharacterTextSplitter**       | 0.58              | 0.42           | 0.52             | 0.44         |
|                                 | **RecursiveCharacterTextSplitter**| 0.44              | 0.47           | 0.60             | 0.52         |
|                                 | **Semantic Chunking**  | 0.30              | 0.34           | 0.46             | 0.48         |
| **15 Specific QA Pairs (Attention)** | **CharacterTextSplitter**   | 0.52              | 0.55           | 0.48             | 0.50         |
|                                 | **RecursiveCharacterTextSplitter**| 0.67              | 0.59           | 0.63             | 0.65         |
|                                 | **Semantic Chunking**  | 0.47              | 0.51           | 0.42             | 0.44         |

Issues:
- StuffingChain (Prompt, Embeddings, LLM)
- CharacterTextSplitter cuts off words
- Semantic Chunking has high variance

Combination of 100 Specific QA Pairs and 15 Specific QA Pairs (Attention) with RecursiveCharSplitter:

| RAG Approach           | Context Precision | Context Recall | Answer Relevancy | Faithfulness |
|------------------------|-------------------|----------------|------------------|--------------|
| **Stuffing**            | 0.45              | 0.42           | 0.58             | 0.58         |
| **Reducing**            | 0.41              | 0.47           | 0.39             | 0.48         |
| **Reranking**           | 0.73              | 0.68           | 0.50             | 0.54         |
| **HyDE**                | 0.36              | 0.35           | 0.43             | 0.41         |

Issues:
- LLM
- ReducingChain (Prompt for collapse and combination)
- RerankingChain (Cross-Encoder Model, Retrieval)
- HyDE(Prompt)
