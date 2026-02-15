# -*- coding: utf-8 -*-
"""
Core pipeline functions: Baseline LLM, Naive RAG, and Rerank RAG.
"""
import pandas as pd
from tqdm import tqdm
import time
from langchain.prompts import PromptTemplate

from config import Config
from llm_services import get_llm_response, load_rerankers, rerank_documents

def run_baseline_llm_pipeline(queries_df):
    """
    Pipeline 1: Generates answers using only LLMs without any retrieval.
    
    Iterates through each query and sends it directly to the configured LLMs.
    """
    results = []
    
    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Running Baseline LLM Pipeline"):
        result_row = {
            "Query ID": row["Query ID"],
            "Query": row["Query"],
            "Ground Truth": row["Ground Truth"],
            "Relevant chunks ID": row["Relevant chunks ID"]
        }
        
        # The prompt for the baseline model is simple, just providing a role and the question.
        prompt = f"Role: You are an expert in building retro-commissioning (RCx).\nQuestion: {row['Query']}\nAnswer:"

        for model_name in Config.ALL_LLMS:
            answer = get_llm_response(prompt, model_name)
            # Use a simple name for the column header
            clean_model_name = model_name.split('/')[-1]
            result_row[f"Response_{clean_model_name}"] = answer
            time.sleep(1)  # Add a small delay to avoid overwhelming APIs

        results.append(result_row)

    df = pd.DataFrame(results)
    df.to_csv(Config.LLM_RESPONSE_FILE, index=False)
    print(f"Baseline LLM responses saved to {Config.LLM_RESPONSE_FILE}")
    return df

def run_rerank_rag_pipeline(queries_df, vector_store):
    """
    Pipeline 3: An advanced RAG pipeline using a reranker.
    
    First, it retrieves a larger set of documents (top-n). Then, for each
    configured reranker model, it re-sorts these documents and selects the
    new top-k to use as context for the LLM.
    """
    results = []
    # Retrieve more documents initially for the reranker to work with
    retriever = vector_store.as_retriever(search_kwargs={"k": Config.RERANK_RETRIEVAL_TOP_K})
    prompt_template = PromptTemplate.from_template(Config.PROMPT_TEMPLATE)
    rerankers = load_rerankers()

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Running Rerank RAG Pipeline"):
        query = row["Query"]
        initial_docs = retriever.invoke(query)
        
        for reranker_name, reranker_model in rerankers.items():
            print(f"\n  Reranking with {reranker_name} for Query ID {row['Query ID']}...")
            reranked_docs = rerank_documents(query, initial_docs, reranker_model)
            # Select the top documents after reranking
            top_reranked_docs = reranked_docs[:Config.RETRIEVAL_TOP_K]
            
            context = "\n\n".join([f"Source: {doc.metadata['source']}; Chunk ID: {doc.metadata['chunk_id']}\nContent: {doc.page_content}" for doc in top_reranked_docs])
            reranked_ids = [doc.metadata['chunk_id'] for doc in top_reranked_docs]

            result_row = {
                "Query ID": row["Query ID"],
                "Query": query,
                "Ground Truth": row["Ground Truth"],
                "Relevant chunks ID": row["Relevant chunks ID"],
                "Reranker": reranker_name,
                "Retrieved chunks ID": reranked_ids
            }
            
            prompt = prompt_template.format(context=context, question=query)
            
            for llm_name in Config.ALL_LLMS:
                answer = get_llm_response(prompt, llm_name)
                clean_llm_name = llm_name.split('/')[-1]
                result_row[f"Response_{clean_llm_name}"] = answer
                time.sleep(1)

            results.append(result_row)

    df = pd.DataFrame(results)
    df.to_csv(Config.RERANK_RAG_FILE, index=False)
    print(f"Rerank RAG responses saved to {Config.RERANK_RAG_FILE}")
    return df
