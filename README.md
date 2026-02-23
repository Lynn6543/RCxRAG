# RCxRAG: A Retrieval-Augmented Generation Model for Automated Information Access in Building Retro-Commissioning
Retro-commissioning (RCx) is a cost-effective measure to improve the performance of existing buildings. However, its wider implementation is often hindered by fragmented RCx knowledge that is difficult for practitioners to access. Although large language models (LLMs) are widely available, they lack structured and domain-specific RCx knowledge to provide reliable support. To address this issue, this study developed a domain-specific retrieval-augmented generation (RAG) model, termed RCxRAG, to improve RCx knowledge management. Following a multi-stage design science methodology, RCxRAG incorporates a re-ranking mechanism in the retrieval module and utilizes the text generation abilities of LLMs to transform static RCx documents into a query-driven knowledge system. Empirical results show that RCxRAG significantly improves retrieval performance compared to naïve RAG. Both naïve RAG and RCxRAG generate higher-quality responses than the general LLM, with the re-ranking mechanism in RCxRAG further enhancing answer quality. Human expert evaluations confirm that RCxRAG achieves the best performance in correctness, usefulness, and fluency, followed by the naïve RAG model. These findings highlight the potential of RCxRAG as an intelligent tool for improving information access, enhancing data-driven decision-making in RCx knowledge management.

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) pipeline designed for RCx knowledge retrieval and evaluation. The project includes data loading, pipeline construction, LLM service integration, and performance evaluation modules.

## RCxRAG Overview 
<img width="450" height="761" alt="d6577b97-02d9-4d09-b470-a7a003409d01" src="https://github.com/user-attachments/assets/bb96b3de-d8bb-4ce1-88c3-84ca5fd91152" />

## Quick Start
### Installation
```python  
pip install -r requirements.txt  
```


