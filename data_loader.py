# -*- coding: utf-8 -*-
"""
Functions for loading documents and queries.
"""
import os
import re
import pandas as pd
from tqdm import tqdm
from langchain.docstore.document import Document
from config import Config

def load_documents(path=Config.CHUNK_DIR):
    """
    Loads documents from a directory, parsing metadata from the file content.
    Each document's metadata (source, category, title, chunk_id) is extracted
    from a YAML-like header in the markdown file.
    """
    documents = []
    filenames = [f for f in os.listdir(path) if f.endswith('.md')]
    print(f"Loading {len(filenames)} documents from '{path}'...")
    
    for filename in tqdm(filenames, desc="Loading document chunks"):
        file_path = os.path.join(path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Use regex to find metadata. group(1) captures the value.
                category_match = re.search(r"Category: (.*)", content)
                source_match = re.search(r"Source: (.*)", content)
                title_match = re.search(r"Title: (.*)", content)
                chunk_id_match = re.search(r"Chunk ID: (.*)", content)

                if not all([category_match, source_match, title_match, chunk_id_match]):
                    print(f"Warning: Could not parse all metadata for file {filename}. Skipping.")
                    continue

                category = category_match.group(1).strip()
                source = source_match.group(1).strip()
                title = title_match.group(1).strip()
                chunk_id = chunk_id_match.group(1).strip()
                
                # The actual page content is after the '---' header separator
                page_content = content.split('---', 2)[-1].strip()
                
                metadata = {
                    "source": source,
                    "category": category,
                    "title": title,
                    "chunk_id": chunk_id,
                    "filename": filename
                }
                documents.append(Document(page_content=page_content, metadata=metadata))
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
            
    return documents

def load_queries(file_path=Config.QUERY_FILE):
    """
    Loads queries from a CSV file into a pandas DataFrame.
    The 'Relevant chunks ID' column is converted from a string to a list of strings.
    """
    print(f"Loading queries from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # Clean up and convert 'Relevant chunks ID'
        df['Relevant chunks ID'] = df['Relevant chunks ID'].apply(
            lambda x: [item.strip() for item in str(x).split(',') if item.strip()]
        )
        return df
    except FileNotFoundError:
        print(f"Error: Query file not found at {file_path}")
        return pd.DataFrame()

