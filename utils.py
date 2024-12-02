"""Utility functions for data processing and index creation."""

import os
from typing import List, Optional, Generator
from pathlib import Path
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import streamlit as st
import faiss
import numpy as np
import torch
from tqdm import tqdm
import gc
from PyPDF2 import PdfReader
import glob

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine, SQLAutoVectorQueryEngine

from config import (
    DATA_DIR,
    DOCS_DIR,
    STORAGE_DIR,
    DB_PATH,
    DB_URL,
    REQUIRED_COLUMNS,
    EMBEDDING_DIMENSION,
    DEVICE,
    BATCH_SIZE,
    MAX_LENGTH,
)

def batch_generator(data: list, batch_size: int) -> Generator:
    """Generate batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def save_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, directory: str) -> str:
    """Save an uploaded file to the specified directory."""
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def preprocess_csv(csv_path: str) -> str:
    """Preprocess CSV file with memory optimization."""
    try:
        # Process in small chunks
        chunk_size = 1000  # Reduced chunk size
        chunks = []
        
        # Get total rows first
        total_rows = sum(1 for _ in open(csv_path)) - 1
        
        with st.progress("Processing CSV data"):
            for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
                # Update progress
                progress = (i * chunk_size) / total_rows
                st.progress(min(progress, 1.0))
                
                # Process financial columns
                for column in chunk.columns:
                    if any(x in column.lower() for x in ['amount', 'balance', 'debit', 'credit']):
                        chunk[column] = pd.to_numeric(
                            chunk[column].astype(str)
                            .str.replace(r'[\$,)]', '', regex=True)
                            .str.replace(r'[(]', '-', regex=True),
                            errors='coerce'
                        )
                    
                    elif any(x in column.lower() for x in ['date', 'period']):
                        chunk[column] = pd.to_datetime(chunk[column], errors='coerce')
                    
                    elif any(x in column.lower() for x in ['account', 'gl']):
                        chunk[column] = chunk[column].astype(str).str.strip()
                
                # Handle missing values
                chunk = chunk.fillna({
                    col: 0.0 if any(x in col.lower() for x in ['amount', 'balance', 'debit', 'credit'])
                    else 'N/A' for col in chunk.columns
                })
                
                chunks.append(chunk)
                
                # Free memory
                del chunk
                gc.collect()
        
        # Combine chunks efficiently
        output_path = os.path.join(DATA_DIR, 'processed_data.csv')
        
        # Write first chunk with headers
        chunks[0].to_csv(output_path, index=False, mode='w')
        
        # Append remaining chunks without headers
        for chunk in chunks[1:]:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
            del chunk
        
        gc.collect()
        return output_path
        
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        raise

def process_pdf(pdf_path: str) -> None:
    """Process PDF with memory optimization."""
    try:
        # Load PDF in chunks
        reader = PdfReader(pdf_path)
        texts = []
        
        with st.progress("Processing PDF"):
            for i, page in enumerate(reader.pages):
                # Update progress
                st.progress((i + 1) / len(reader.pages))
                
                # Extract text
                text = page.extract_text()
                if text.strip():
                    texts.append(text)
                
                # Free memory
                del page
                gc.collect()
        
        # Save processed text
        output_path = os.path.join(DOCS_DIR, os.path.basename(pdf_path).replace('.pdf', '.txt'))
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n\n')
                del text
        
        gc.collect()
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        raise

def build_sql_index() -> NLSQLTableQueryEngine:
    """Build SQL index with memory optimization."""
    try:
        # Create database connection
        engine = create_engine(DB_URL)
        
        # Process in small chunks
        chunk_size = 1000
        csv_path = os.path.join(DATA_DIR, 'processed_data.csv')
        
        with st.progress("Building SQL index"):
            total_rows = sum(1 for _ in open(csv_path)) - 1
            
            for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
                # Update progress
                progress = (i * chunk_size) / total_rows
                st.progress(min(progress, 1.0))
                
                # Write to database
                chunk.to_sql('gl_data', engine, if_exists='append' if i > 0 else 'replace', index=False)
                del chunk
                gc.collect()
        
        # Create query engine
        sql_db = SQLDatabase(engine, include_tables=['gl_data'])
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_db,
            tables=['gl_data'],
        )
        
        return query_engine
        
    except Exception as e:
        st.error(f"Error building SQL index: {str(e)}")
        raise

def build_vector_index() -> VectorStoreIndex:
    """Build vector index with memory optimization."""
    try:
        documents = []
        
        # Process text files in chunks
        for txt_file in glob.glob(os.path.join(DOCS_DIR, '*.txt')):
            with open(txt_file, 'r', encoding='utf-8') as f:
                chunk_size = 1000  # characters
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    if chunk.strip():
                        documents.append(Document(text=chunk))
                    
                    # Free memory
                    del chunk
                    gc.collect()
        
        # Build index in batches
        storage_context = StorageContext.from_defaults()
        
        with st.progress("Building vector index"):
            for i, doc in enumerate(documents):
                # Update progress
                st.progress((i + 1) / len(documents))
                
                # Add to index
                index = VectorStoreIndex.from_documents(
                    [doc],
                    storage_context=storage_context,
                    show_progress=False
                )
                
                # Free memory
                del doc
                gc.collect()
        
        return index
        
    except Exception as e:
        st.error(f"Error building vector index: {str(e)}")
        raise

def get_sql_tool(sql_query_engine: NLSQLTableQueryEngine) -> QueryEngineTool:
    """Create SQL query tool for financial data."""
    return QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for analyzing financial and GL data through SQL queries. "
            "Can handle questions about account balances, transaction details, "
            "journal entries, and financial metrics. Supports analysis of debits, "
            "credits, account relationships, and transaction patterns."
        ),
    )

def get_vector_tool(index: VectorStoreIndex) -> QueryEngineTool:
    """Create vector query tool for financial documentation."""
    vector_store_info = VectorStoreInfo(
        content_info="Contains financial documentation including GL procedures, "
                    "accounting policies, financial controls, audit requirements, "
                    "and reporting standards. Covers GAAP/IFRS guidelines, "
                    "account structures, and financial processes.",
        metadata_info=[
            MetadataInfo(
                name="title",
                type="str",
                description="Financial and GL Documentation"
            ),
        ],
    )
    
    vector_auto_retriever = VectorIndexAutoRetriever(index, vector_store_info=vector_store_info)
    retriever_query_engine = RetrieverQueryEngine.from_args(
        vector_auto_retriever,
        node_postprocessors=[],  # Minimize memory usage
    )
    
    return QueryEngineTool.from_defaults(
        query_engine=retriever_query_engine,
        description="Useful for answering questions about financial procedures, "
                   "accounting policies, GL structures, and compliance requirements."
    )

@torch.no_grad()  # Disable gradient computation to save memory
def create_query_engine(sql_tool: QueryEngineTool, vector_tool: QueryEngineTool) -> SQLAutoVectorQueryEngine:
    """Create combined query engine from SQL and vector tools."""
    return SQLAutoVectorQueryEngine(sql_tool, vector_tool)
