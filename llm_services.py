# -*- coding: utf-8 -*-
"""
Functions for interacting with LLM APIs (OpenAI, Hugging Face) and rerankers.
"""
import requests
from langchain_openai import ChatOpenAI
from sentence_transformers.cross_encoder import CrossEncoder
from config import Config

def get_openai_completion(prompt, model_name):
    """
    Gets a completion from an OpenAI model.
    Handles potential API errors.
    """
    try:
        chat = ChatOpenAI(
            model_name=model_name,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            api_key=Config.OPENAI_API_KEY
        )
        response = chat.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error calling OpenAI API for model {model_name}: {e}")
        return "Error: Could not get response from OpenAI."

# def get_hf_completion(prompt, model_name):
#     """
#     Gets a completion from a Hugging Face Serverless Inference API model.
#     Handles potential API errors and timeouts.
#     """
#     api_url = f"https://router.huggingface.co/models/{model_name}"
#     headers = {"Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}"}
    
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": Config.LLM_MAX_TOKENS,
#             "temperature": Config.LLM_TEMPERATURE if Config.LLM_TEMPERATURE > 0 else 0.01, # Temp 0 is not always supported
#             "return_full_text": False
#         }
#     }
    
#     try:
#         response = requests.post(api_url, headers=headers, json=payload, timeout=180) # Increased timeout
#         response.raise_for_status()
#         # The response is typically a list with a dictionary
#         generated_text = response.json()[0]['generated_text']
#         return generated_text.strip()
#     except requests.exceptions.RequestException as e:
#         print(f"Error calling HF Inference API for model {model_name}: {e}")
#         return f"Error: API request failed for {model_name}."
#     except (KeyError, IndexError) as e:
#         print(f"Error processing response from HF model {model_name}: Unexpected format. {e}")
#         print(f"Full response: {response.text}")
#         return f"Error: Could not parse response from {model_name}."
#     except Exception as e:
#         print(f"An unexpected error occurred with HF model {model_name}: {e}")
#         return f"Error: An unexpected error occurred with {model_name}."



def get_hf_completion(prompt, model_name):
    """
    Gets a completion from Hugging Face via the Router (OpenAI-compatible API).
    """
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,  # e.g. "Qwen/Qwen2.5-7B-Instruct"
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(Config.LLM_TEMPERATURE) if Config.LLM_TEMPERATURE and Config.LLM_TEMPERATURE > 0 else 0.01,
        "max_tokens": int(Config.LLM_MAX_TOKENS),
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)
        if response.status_code != 200:
            print("HF status:", response.status_code)
            print("HF body:", response.text[:2000])
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        print(f"Error calling HF Router for model {model_name}: {e}")
        return f"Error: API request failed for {model_name}."
    except (KeyError, IndexError) as e:
        print(f"Error processing response from HF model {model_name}: Unexpected format. {e}")
        print(f"Full response: {response.text}")
        return f"Error: Could not parse response from {model_name}."


def get_llm_response(prompt, model_name):
    """
    Unified function to call any configured LLM.
    
    Args:
        prompt (str): The input prompt for the LLM.
        model_name (str): The identifier of the model to use.

    Returns:
        str: The response from the language model.
    """
    print(f"  Querying {model_name}...")
    if model_name == Config.OPENAI_LLM:
        return get_openai_completion(prompt, model_name)
    elif model_name in Config.HF_LLMS.values():
        return get_hf_completion(prompt, model_name)
    else:
        print(f"Warning: Unknown model '{model_name}'.")
        return "Error: Unknown model."

def load_rerankers():
    """Loads and returns a dictionary of CrossEncoder reranking models."""
    print("Loading reranker models...")
    rerankers = {
        name: CrossEncoder(path) 
        for name, path in Config.RERANKER_MODELS.items()
    }
    print("Reranker models loaded.")
    return rerankers

def rerank_documents(query, documents, reranker_model):
    """
    Reranks a list of documents based on a query using a CrossEncoder model.

    Args:
        query (str): The user query.
        documents (list[Document]): The list of documents to be reranked.
        reranker_model (CrossEncoder): The loaded CrossEncoder model.

    Returns:
        list[Document]: The list of documents, sorted by reranking score.
    """
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker_model.predict(pairs)
    
    # Combine documents with scores and sort in descending order
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    # Return just the sorted documents
    return [doc for _, doc in scored_docs]
