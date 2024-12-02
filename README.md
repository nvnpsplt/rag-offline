# Risk Framework Assistant

A Streamlit-based application for analyzing risk framework documentation and transaction data using RAG (Retrieval-Augmented Generation).

## Features

- Process and analyze transaction data from CSV files
- Extract information from PDF documentation
- Query both structured (SQL) and unstructured (Vector) data
- Memory-efficient processing of large datasets
- GPU acceleration support
- Interactive chat interface

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for GPU acceleration)
- 16GB+ RAM
- 4GB+ GPU Memory (if using GPU)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Upload your data:
   - Upload CSV files containing transaction data
   - Upload PDF documentation about the risk framework

3. Initialize the query engine using the button in the interface

4. Use the chat interface to:
   - Query transaction data
   - Ask questions about the risk framework
   - Analyze risk scores and deviations

## Data Format

### CSV Requirements
The CSV files should contain the following columns:
- ACCOUNTDOCID
- BLENDED_RISK_SCORE
- AI_RISK_SCORE
- STAT_SCORE
- RULES_RISK_SCORE
- CONTROL_DEVIATION
- MONITORING_DEVIATION

### PDF Requirements
The PDF should contain documentation about:
- Risk framework overview
- Rules framework
- Statistical framework
- AI framework

## Memory Management

The application includes several memory optimization features:
- Batch processing for large files
- GPU memory management
- Efficient data chunking
- Mixed precision (FP16) support for GPU operations

## Troubleshooting

1. GPU Memory Issues:
   - The application will automatically fall back to CPU if GPU memory is insufficient
   - Adjust batch sizes in config.py if needed
   - Clear GPU cache using torch.cuda.empty_cache()

2. CPU Memory Issues:
   - Reduce batch sizes in config.py
   - Process smaller chunks of data at a time
   - Close other memory-intensive applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
