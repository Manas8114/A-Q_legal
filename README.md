# ⚖️ Legal QA System

A comprehensive AI-powered legal question-answering system that combines advanced retrieval and generation techniques for accurate legal information extraction. Built with modern AI technologies including Gemini API, BERT, and hybrid retrieval systems.

## 🚀 Key Features

- **🤖 Dual AI Models**: Choose between Extractive (BERT-based) or Generative (Gemini API) answering
- **📚 Comprehensive Legal Data**: 3,000+ Q&A pairs covering Constitution, CrPC, and IPC
- **⚡ Optimized Performance**: Pre-trained models for instant loading and fast responses
- **🎯 Smart Retrieval**: Hybrid BM25 + Dense retrieval with FAISS indexing
- **📊 Question Classification**: Bayesian classifier for legal question categorization
- **💾 Intelligent Caching**: Smart answer caching for improved performance
- **🌐 Modern Web Interface**: Streamlit-based GUI with model selection options
- **🔧 Model Selection**: Choose extractive, generative, or hybrid approaches

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Question      │───▶│  Classification  │───▶│  Similarity     │
│   Input         │    │  (Bayesian)      │    │  Check          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Answer        │◀───│  Answer Ranking  │◀───│  Context        │
│   Output        │    │  (Hybrid)        │    │  Retrieval      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Answer          │
                       │  Generation      │
                       │  (Extractive +   │
                       │   Generative)    │
                       └──────────────────┘
```

## 📦 Quick Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd A-Qlegal-main

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🎯 Usage

### 1. 🚀 Quick Start (Recommended)
```bash
# Launch the optimized GUI with pre-trained models
streamlit run optimized_gui.py --server.port 8504
```
Open `http://localhost:8504` in your browser and start asking legal questions!

### 2. 🔧 Load Saved System (Fast Loading)
```bash
# Load the pre-trained system (loads in seconds)
python load_saved_system.py
```

### 3. 💾 Save System State
```bash
# Save the current system state
python save_system.py
```

### 4. 🌐 Web Interface Options

#### Optimized GUI (Recommended)
- **URL**: `http://localhost:8504`
- **Features**: Model selection, fast loading, modern UI
- **Launch**: `streamlit run optimized_gui.py --server.port 8504`

#### API Server
```bash
# Start FastAPI server
python -m src.api.main
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 🎨 Model Selection Options

The system offers three answering modes:

### 1. 🔍 Extractive Only
- **Model**: BERT-based extractive model
- **Best for**: Direct answer extraction from legal texts
- **Speed**: Very fast
- **Use case**: When you need precise quotes from legal documents

### 2. 🤖 Generative Only (Gemini)
- **Model**: Google Gemini 1.5 Flash API
- **Best for**: Comprehensive, contextual answers
- **Speed**: Fast (API-based)
- **Use case**: When you need detailed explanations and analysis

### 3. 🔄 Both (Hybrid)
- **Models**: Combines both extractive and generative approaches
- **Best for**: Most comprehensive answers
- **Speed**: Moderate
- **Use case**: When you want the best of both worlds

## 📁 Project Structure

```
A-Qlegal-main/
├── src/                          # Core system modules
│   ├── main.py                   # Main system orchestrator
│   ├── data/                     # Data processing
│   │   ├── dataset_loader.py    # Dataset loading utilities
│   │   ├── embeddings.py         # Embedding generation
│   │   └── preprocessor.py       # Text preprocessing
│   ├── classification/           # Question classification
│   │   ├── bayesian_classifier.py
│   │   └── syntactic_features.py
│   ├── retrieval/                # Retrieval systems
│   │   ├── bm25_retriever.py     # BM25 lexical retrieval
│   │   ├── dense_retriever.py    # Dense semantic retrieval
│   │   └── hybrid_retriever.py   # Hybrid retrieval system
│   ├── generation/               # Answer generation
│   │   ├── extractive_model.py   # BERT-based extraction
│   │   ├── generative_model.py  # T5/Gemini generation
│   │   ├── gemini_model.py      # Gemini API integration
│   │   └── answer_ranker.py     # Answer ranking system
│   ├── utils/                    # Utility functions
│   │   ├── caching.py           # Answer caching
│   │   └── similarity.py        # Similarity computation
│   └── api/                      # FastAPI backend
│       └── main.py              # API endpoints
├── data/                         # Legal datasets
│   ├── constitution_qa.json      # Constitution Q&A pairs
│   ├── crpc_qa.json             # CrPC Q&A pairs
│   └── ipc_qa.json              # IPC Q&A pairs
├── models/                       # Saved model files
│   ├── optimized_legal_qa_*      # Optimized system models
│   └── answer_cache.pkl         # Answer cache
├── optimized_gui.py              # Main GUI application
├── load_saved_system.py          # System loading script
├── save_system.py               # System saving script
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 📊 Dataset Information

The system includes comprehensive legal datasets:

- **Constitution**: 1,022 Q&A pairs covering fundamental rights, duties, and constitutional provisions
- **CrPC**: 1,010 Q&A pairs covering criminal procedure code
- **IPC**: 1,010 Q&A pairs covering Indian penal code
- **Total**: 3,042 Q&A pairs across 3 legal domains
- **Categories**: Fact-based, procedural, and interpretive questions

## 🔧 Configuration

### Basic Configuration
```python
config = {
    'data_dir': 'data/',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'extractive_model': 'bert-base-uncased',
    'generative_model': 'gemini-1.5-flash',
    'gemini_api_key': 'your-gemini-api-key',
    'question_categories': ['fact', 'procedure', 'interpretive'],
    'bm25_weight': 0.3,
    'dense_weight': 0.7,
    'default_top_k': 15,
    'max_context_length': 2000,
    'similarity_threshold': 0.8,
    'confidence_threshold': 0.7
}
```

### Gemini API Setup
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## 🚀 Performance Features

### ⚡ Optimized Loading
- **Pre-trained Models**: System loads in seconds instead of minutes
- **Smart Caching**: Intelligent answer caching for similar questions
- **Efficient Retrieval**: FAISS-based similarity search
- **Batch Processing**: Optimized batch operations

### 🎯 Accuracy Improvements
- **Hybrid Retrieval**: Combines lexical (BM25) and semantic (dense) retrieval
- **Question Classification**: Bayesian classifier for better context understanding
- **Answer Ranking**: Multi-factor ranking system for best answer selection
- **Confidence Scoring**: Detailed confidence metrics for answer reliability

## 📈 Example Usage

### Python API
```python
from src.main import LegalQASystem

# Initialize system
config = {
    'gemini_api_key': 'your-api-key',
    'default_top_k': 15,
    'max_context_length': 2000
}
system = LegalQASystem(config)

# Load datasets
dataset_paths = {
    'constitution': 'data/constitution_qa.json',
    'crpc': 'data/crpc_qa.json',
    'ipc': 'data/ipc_qa.json'
}
system.initialize_system(dataset_paths)

# Ask questions with different models
# Extractive only
result = system.ask_question("What is the punishment for theft?", use_extractive_only=True)

# Generative only (Gemini)
result = system.ask_question("Explain the fundamental rights", use_generative_only=True)

# Both (Hybrid)
result = system.ask_question("What are the procedures for filing a case?", use_extractive_only=False, use_generative_only=False)
```

### REST API
```bash
# Ask a question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the fundamental right to equality?",
       "use_generative_only": true,
       "top_k": 5
     }'

# Get system status
curl -X GET "http://localhost:8000/status"
```

## 🎨 Web Interface Features

### Optimized GUI (`optimized_gui.py`)
- **Model Selection**: Choose between Extractive, Generative, or Hybrid modes
- **Real-time Status**: Live system status and model information
- **Sample Questions**: Pre-loaded example questions for testing
- **Answer Display**: Formatted answers with confidence indicators
- **Fast Loading**: Pre-trained models load in seconds
- **Modern UI**: Clean, responsive design with Streamlit

### Key Features
- ✅ **Model Selection Interface**: Easy switching between answering modes
- ✅ **System Status Dashboard**: Real-time monitoring of all components
- ✅ **Sample Questions**: Quick access to test questions
- ✅ **Answer Formatting**: Clean, readable answer display
- ✅ **Confidence Indicators**: Visual confidence scoring
- ✅ **Performance Metrics**: Response time and accuracy tracking

## 🔬 Advanced Features

### 1. Hybrid Retrieval System
- **BM25**: Lexical retrieval for exact keyword matching
- **Dense Retrieval**: Semantic retrieval using sentence transformers
- **FAISS Indexing**: Efficient similarity search
- **Configurable Weights**: Adjustable balance between retrieval methods

### 2. Question Classification
- **Bayesian Classifier**: Probabilistic classification of legal questions
- **Syntactic Features**: spaCy-based linguistic feature extraction
- **Categories**: Fact-based, procedural, and interpretive questions
- **Confidence Scoring**: Classification confidence metrics

### 3. Answer Generation
- **Extractive Model**: BERT-based span extraction from retrieved contexts
- **Generative Model**: Gemini API for comprehensive answer generation
- **Answer Ranking**: Multi-factor ranking considering relevance, confidence, and source
- **Context Truncation**: Smart context management for optimal performance

### 4. Caching System
- **Similarity Detection**: Tanimoto coefficient for near-duplicate questions
- **Intelligent Caching**: Automatic caching of similar questions
- **Cache Management**: Efficient cache storage and retrieval
- **Performance Boost**: Significant speed improvements for repeated queries

## 🚀 Getting Started Examples

### Example 1: Basic Question
```python
# What are the fundamental rights in the Constitution?
# Expected: Comprehensive list of fundamental rights from Part III
```

### Example 2: Procedural Question
```python
# What is the procedure for filing a criminal case?
# Expected: Step-by-step procedure from CrPC
```

### Example 3: IPC Question
```python
# What is the punishment for theft under IPC?
# Expected: Specific punishment details from IPC sections
```

## 🔧 Troubleshooting

### Common Issues

1. **Gemini API Error**
   ```bash
   # Solution: Check API key and internet connection
   export GEMINI_API_KEY="your-valid-api-key"
   ```

2. **Model Loading Issues**
   ```bash
   # Solution: Reinstall dependencies
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Memory Issues**
   ```bash
   # Solution: Use optimized configuration
   # The system automatically uses optimized settings for better performance
   ```

## 📊 System Status

### Current Status
- ✅ **Classifier**: Trained and ready
- ✅ **Extractive Model**: Trained and ready  
- ✅ **Generative Model**: Gemini API integrated
- ✅ **Retrieval System**: Hybrid BM25 + Dense retrieval
- ✅ **Datasets**: 3,042 Q&A pairs loaded
- ✅ **Caching**: Intelligent answer caching active

### Performance Metrics
- **Loading Time**: ~10-15 seconds (with saved models)
- **Response Time**: 1-3 seconds per question
- **Accuracy**: High accuracy across all legal domains
- **Memory Usage**: Optimized for efficient operation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini API**: For advanced generative capabilities
- **Hugging Face**: For transformer models and utilities
- **spaCy**: For natural language processing
- **FAISS**: For efficient similarity search
- **Streamlit**: For the web interface
- **FastAPI**: For the backend API

## 📞 Support

For questions, issues, or contributions:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Contact the maintainers

---

**⚖️ Legal QA System** - Powered by AI for comprehensive legal question answering

*Built with ❤️ for the legal community*# -Question-Answering-legal
