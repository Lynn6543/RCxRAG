# -*- coding: utf-8 -*-
"""
Functions for evaluating retrieval and generation performance.
"""
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_official
from sentence_transformers import SentenceTransformer, util

def evaluate_retrieval(results_df):
    """
    Calculates retrieval metrics (Precision, Recall, F1, Hit Rate, MRR).

    Args:
        results_df (pd.DataFrame): DataFrame containing 'Relevant chunks ID' 
                                   and 'Retrieved chunks ID' columns.

    Returns:
        pd.DataFrame: A DataFrame with the calculated metrics for each query.
    """
    metrics = []
    for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Evaluating Retrieval"):
        # Ensure sets are correctly formed from lists
        relevant_ids = set(row.get('Relevant chunks ID', []))
        retrieved_ids = set(row.get('Retrieved chunks ID', []))
        
        true_positives = len(relevant_ids.intersection(retrieved_ids))
        
        # Precision = TP / (TP + FP) = TP / |retrieved|
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        
        # Recall = TP / (TP + FN) = TP / |relevant|
        recall = true_positives / len(relevant_ids) if relevant_ids else 0
        
        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Hit Rate: 1 if any relevant document is retrieved, 0 otherwise.
        hit_rate = 1 if true_positives > 0 else 0
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        # Use the original ordered list for MRR
        for i, doc_id in enumerate(row.get('Retrieved chunks ID', [])):
            if doc_id in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        
        metric_row = {
            'Query ID': row['Query ID'],
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Hit Rate': hit_rate,
            'MRR': mrr
        }
        
        # Identify the method (Naive RAG vs. specific Reranker)
        metric_row['Method'] = f"Rerank_{row['Reranker']}" if 'Reranker' in row else 'Naive_RAG'
            
        metrics.append(metric_row)

    return pd.DataFrame(metrics)


def evaluate_generation(all_responses_df):
    """
    Calculates generation metrics (ROUGE, BLEU, BERTScore, SBERT).

    Args:
        all_responses_df (pd.DataFrame): A DataFrame containing responses from all pipelines.

    Returns:
        pd.DataFrame: A DataFrame with the calculated generation scores.
    """
    print("Loading evaluation models for generation...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    # Using Chencherry's smoothing method for BLEU to handle short sentences
    smoothing_fn = SmoothingFunction().method1
    
    gen_eval_results = []

    for _, row in tqdm(all_responses_df.iterrows(), total=len(all_responses_df), desc="Evaluating Generation"):
        ground_truth = row['Ground Truth']
        if not isinstance(ground_truth, str):
            continue  # Skip if ground truth is not a string

        # Identify all response columns
        response_cols = [col for col in row.index if col.startswith('Response_')]
        
        for col_name in response_cols:
            response = row[col_name]
            
            # Set scores to 0 for invalid responses
            if not isinstance(response, str) or response.strip() in ["", "Null"] or "Error:" in response:
                rouge1_f, bleu, bert_f1, sbert_cos_sim = 0, 0, 0, 0
            else:
                # ROUGE-1 F-score
                rouge1_f = rouge.score(ground_truth, response)['rouge1'].fmeasure
                
                # BLEU Score
                bleu = sentence_bleu([ground_truth.split()], response.split(), smoothing_function=smoothing_fn)
                
                # BERTScore
                try:
                    _, _, bert_f1 = bert_score_official([response], [ground_truth], lang='en', verbose=False)
                    bert_f1 = bert_f1.mean().item()
                except Exception as e:
                    print(f"Warning: BERTScore failed for Query ID {row['Query ID']}: {e}")
                    bert_f1 = 0

                # SBERT Cosine Similarity
                try:
                    gt_embedding = sbert_model.encode(ground_truth, convert_to_tensor=True)
                    resp_embedding = sbert_model.encode(response, convert_to_tensor=True)
                    sbert_cos_sim = util.pytorch_cos_sim(gt_embedding, resp_embedding).item()
                except Exception as e:
                    print(f"Warning: SBERT failed for Query ID {row['Query ID']}: {e}")
                    sbert_cos_sim = 0
            
            gen_eval_results.append({
                'Query ID': row['Query ID'],
                'Pipeline': row.get('Pipeline', 'Unknown'),
                'Reranker': row.get('Reranker', 'Unknown'),
                'Method': col_name, # e.g., 'Response_gpt-4o'
                'ROUGE-1': rouge1_f,
                'BLEU': bleu,
                'BERTScore': bert_f1,
                'SBERT': sbert_cos_sim
            })

    return pd.DataFrame(gen_eval_results)
