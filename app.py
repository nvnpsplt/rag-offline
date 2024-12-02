"""Streamlit application for GL transaction risk analysis using RAG."""

import streamlit as st
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import psutil
import os

from utils import (
    save_uploaded_file,
    preprocess_csv,
    process_pdf,
    build_sql_index,
    build_vector_index,
    get_sql_tool,
    get_vector_tool,
    create_query_engine,
)
from config import DATA_DIR, DOCS_DIR, DEVICE, MAX_LENGTH

# Model configurations - Using ultra lightweight models
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Only 1.1B parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Tiny embedding model

SYSTEM_PROMPT = """You are a GL Transaction Analysis Assistant. Analyze GL transactions and explain why they are flagged as suspicious based on the data and documentation provided.

Focus on:
1. Transaction Risk Factors:
   - Unusual amounts or patterns
   - Suspicious account combinations
   - Timing irregularities
   - Policy violations

2. Analysis Steps:
   - State transaction details
   - List risk indicators
   - Cite violated rules/policies
   - Explain suspicious factors
   - Suggest required documentation

Be precise and reference specific criteria from the documentation."""

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024 / 1024  # MB
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        return f"RAM: {ram_usage:.1f}MB, GPU: {gpu_usage:.1f}MB"
    return f"RAM: {ram_usage:.1f}MB"

def clear_memory():
    """Aggressively clear memory."""
    if "llm" in st.session_state:
        del st.session_state["llm"]
    if "embed_model" in st.session_state:
        del st.session_state["embed_model"]
    if "query_engine" in st.session_state:
        del st.session_state["query_engine"]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load embedding model with caching."""
    try:
        return HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
            device=DEVICE,
            embed_batch_size=1,
        )
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def load_llm():
    """Load LLM with caching."""
    try:
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        
        return HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=1024,  # Reduced context window
            max_new_tokens=128,  # Reduced output length
            generate_kwargs={
                "temperature": 0.1,
                "top_p": 0.7,
                "top_k": 10,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            },
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as e:
        st.error(f"Error loading language model: {str(e)}")
        return None

def initialize_models():
    """Initialize models with better error handling."""
    try:
        st.session_state["embed_model"] = load_embedding_model()
        if st.session_state["embed_model"] is None:
            raise ValueError("Failed to load embedding model")
        
        st.session_state["llm"] = load_llm()
        if st.session_state["llm"] is None:
            raise ValueError("Failed to load language model")
        
        Settings.embed_model = st.session_state["embed_model"]
        Settings.llm = st.session_state["llm"]
        
        st.sidebar.success("Models loaded successfully!")
        st.sidebar.info(f"Memory Usage: {get_memory_usage()}")
        
    except Exception as e:
        clear_memory()
        st.error(f"Error initializing models: {str(e)}")
        st.info("Try refreshing the page or check your available memory.")
        raise

def process_files_in_batches(files, batch_size=2):
    """Process files in small batches to manage memory."""
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        for file in batch:
            file_path = save_uploaded_file(file)
            if file.type == "text/csv":
                with st.spinner(f"Processing CSV file {file.name}..."):
                    preprocess_csv(file_path)
            else:
                with st.spinner(f"Processing PDF file {file.name}..."):
                    process_pdf(file_path)
        clear_memory()
        st.sidebar.info(f"Memory Usage: {get_memory_usage()}")

def handle_file_upload():
    """Handle file uploads with batch processing."""
    try:
        uploaded_files = st.file_uploader(
            "Upload GL Transaction CSV and Documentation PDFs",
            type=["csv", "pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            process_files_in_batches(uploaded_files)
            return True
            
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return False

def setup_query_engine():
    """Setup query engine with memory optimization."""
    try:
        if "query_engine" not in st.session_state:
            clear_memory()
            
            with st.spinner("Building SQL index..."):
                sql_query_engine = build_sql_index()
            
            clear_memory()
            
            with st.spinner("Building vector index..."):
                vector_index = build_vector_index()
            
            clear_memory()
            
            sql_tool = get_sql_tool(sql_query_engine)
            vector_tool = get_vector_tool(vector_index)
            
            st.session_state["query_engine"] = create_query_engine(sql_tool, vector_tool)
            st.success("Query engine initialized successfully!")
    except Exception as e:
        st.error(f"Error setting up query engine: {str(e)}")
        st.info("Please try uploading your files again.")

def chat_interface():
    """Memory-efficient chat interface."""
    if "query_engine" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about GL transactions..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Processing..."):
                        response = st.session_state["query_engine"].query(prompt)
                        st.markdown(response.response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response.response}
                        )
                        st.sidebar.info(f"Memory Usage: {get_memory_usage()}")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                finally:
                    clear_memory()
    else:
        st.info("Please upload your files to start the analysis.")

def main():
    """Main application function with error handling."""
    try:
        st.title("GL Transaction Risk Analysis Assistant")
        st.markdown("""
        Analyze GL transactions and identify suspicious patterns:
        
        1. Upload your files:
           - GL transaction data (CSV)
           - Risk assessment documentation (PDF)
        
        2. Ask questions about:
           - Transaction risk factors
           - Policy violations
           - Suspicious patterns
           - Required documentation
        """)
        
        # Display device info
        device_info = f"Using device: {DEVICE.upper()}"
        if DEVICE == "cuda":
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            device_info += f" | GPU Memory: {gpu_mem:.2f} GB"
        st.sidebar.info(device_info)
        
        # Initialize components
        initialize_models()
        if handle_file_upload():
            setup_query_engine()
        chat_interface()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page to restart the application.")

if __name__ == "__main__":
    main()