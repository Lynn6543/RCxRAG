# -*- coding: utf-8 -*-
"""
Main entry point for the RAG Pipeline Evaluation Script.

This script orchestrates the entire process:
1. Loads data (documents and queries).
2. Sets up the vector store.
3. Runs the selected pipelines (or loads existing results).
4. Performs retrieval and generation evaluations.
5. Generates summary reports and visualizations.
"""
import os
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data')

# Import functions from our modules
from config import Config
from data_loader import load_documents, load_queries
from vector_store_manager import create_vector_store
# from pipelines import run_baseline_llm_pipeline, run_naive_rag_pipeline, run_rerank_rag_pipeline
from pipelines import run_baseline_llm_pipeline
from evaluation import evaluate_retrieval, evaluate_generation
from reporting import create_summary_report, plot_retrieval_results, plot_generation_results

def main():
    """Main function to orchestrate the RAG evaluation process."""
    print("Starting RAG pipeline evaluation script...")
    
    # # --- 1. Load Data ---
    # documents = load_documents()
    # queries_df = load_queries()

    # if documents is None or documents == [] or queries_df.empty:
    #     print("Failed to load documents or queries. Exiting.")
    #     return

    # # --- 2. Create Vector Store ---
    # vector_store = create_vector_store(documents)
    
    # # --- 3. Run Pipelines ---
    # # The user can choose which pipelines to run.
    # # NOTE: Running these will take a very long time and incur API costs.
    # # To evaluate existing results, comment out these lines and ensure the CSV files are in the 'results/' directory.
    
    # # run_pipelines = True # <<<< SET TO False TO SKIP RUNNING PIPELINES AND ONLY EVALUATE
    # run_pipelines = False # <<<< SET TO False TO SKIP RUNNING PIPELINES AND ONLY EVALUATE


    # if run_pipelines:
    #     print("\n--- Running Pipeline 1: Baseline LLM ---")
    #     run_baseline_llm_pipeline(queries_df)
        
    #     # print("\n--- Running Pipeline 2: Naive RAG ---")
    #     # run_naive_rag_pipeline(queries_df, vector_store)
        
    #     # print("\n--- Running Pipeline 3: Rerank RAG ---")
    #     # run_rerank_rag_pipeline(queries_df, vector_store)
    # else:
    #     print("\n--- Skipping pipeline runs. Loading existing results for evaluation. ---")

    # # --- 4. Load Results for Evaluation ---
    # print("\n--- Loading pipeline results for evaluation. ---")
    # try:
    #     # Only load results from pipelines that were actually run
    #     baseline_df = pd.read_csv(Config.LLM_RESPONSE_FILE)
    #     naive_rag_df = pd.read_csv(Config.NAIVE_RAG_FILE)
    #     rerank_rag_df = pd.read_csv(Config.RERANK_RAG_FILE)
    #     print("Pipeline results loaded successfully.")
    # except FileNotFoundError as e:
    #     print(f"Error: Could not find a results file: {e}. Please run the pipelines first by setting 'run_pipelines = True'.")
    #     return

    # # --- 5. Run Retrieval Evaluation ---
    # print("\n--- Evaluating Retrieval Performance ---")
    # # The 'eval()' function is used to safely convert the string representation of a list back into a list object.
    # naive_rag_df['Relevant chunks ID'] = naive_rag_df['Relevant chunks ID'].apply(eval)
    # naive_rag_df['Retrieved chunks ID'] = naive_rag_df['Retrieved chunks ID'].apply(eval)
    # rerank_rag_df['Relevant chunks ID'] = rerank_rag_df['Relevant chunks ID'].apply(eval)
    # rerank_rag_df['Retrieved chunks ID'] = rerank_rag_df['Retrieved chunks ID'].apply(eval)
    
    # ret_eval_naive = evaluate_retrieval(naive_rag_df)
    # ret_eval_rerank = evaluate_retrieval(rerank_rag_df)
    
    # retrieval_df = pd.concat([ret_eval_naive, ret_eval_rerank], ignore_index=True)
    # retrieval_df.to_csv(Config.RETRIEVAL_EVAL_FILE, index=False)
    # print(f"Retrieval evaluation results saved to {Config.RETRIEVAL_EVAL_FILE}")

    # --- 6. Run Generation Evaluation ---
    # print("\n--- Evaluating Generation Performance ---")
    # baseline_df['Pipeline'] = 'Baseline_LLM'
    # naive_rag_df['Pipeline'] = 'Naive_RAG'
    # rerank_rag_df['Pipeline'] = 'Rerank_RAG'
    
    # # Evaluate each pipeline separately for generation
    # generation_df_baseline = evaluate_generation(baseline_df)
    # generation_df_naive = evaluate_generation(naive_rag_df)
    # generation_df_rerank = evaluate_generation(rerank_rag_df)
    
    # # Combine all generation evaluation results
    # generation_df = pd.concat([generation_df_baseline, generation_df_naive, generation_df_rerank], ignore_index=True)

    # generation_df.to_csv(Config.GENERATION_EVAL_FILE, index=False)
    # print(f"Generation evaluation results saved to {Config.GENERATION_EVAL_FILE}")

    # --- 7. Summarize and Plot Results ---
    print("\n--- Creating Final Summary and Plots ---")
    retrieval_df = pd.read_csv(Config.RETRIEVAL_EVAL_FILE)
    generation_df = pd.read_csv(Config.GENERATION_EVAL_FILE)
    create_summary_report(retrieval_df, generation_df)
    plot_retrieval_results(retrieval_df)
    plot_generation_results(generation_df)

    print("\nScript finished successfully!")
    print(f"All outputs are saved in the '{Config.OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()
