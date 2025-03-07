<h1>🤖 Smart AI Assistant<h1>

## Project Overview

Smart AI Assistant is an advanced, context-aware conversational AI application that leverages retrieval-augmented generation (RAG), natural language processing, and cutting-edge safety mechanisms to provide intelligent, contextual responses.

## Features

- 🧠 Retrieval-Augmented Generation (RAG)
- 📊 Multi-persona AI interaction
- 🔒 Advanced content safety filtering
- 💾 Intelligent caching mechanism
- 🔍 Semantic document search
- 🚀 Optimized performance with rate limiting

## Prerequisites

- Python 3.8+
- OpenAI API Key
- CUDA-compatible GPU (recommended but optional)

## Project Structure

```
multi-functional-ai-assistant/
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── nlp_tasks.py            # NLP model management
│   ├── retrieval.py            # Document retrieval and embedding
│   ├── safety_utils.py         # Content safety and rate limiting
│   └── llm_abstraction.py      # LLM wrapper
│
├── data/
│   └── sample.txt              # Sample documents for retrieval
│
├── docs/
│   └── report.pdf              # Technical documentation of the project
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_nlp_tasks.py
│   └── test_retrieval.py
│ 
│├── app/
│   ├── main.py                    # Streamlit application
│ 
├── requirements.txt            # Project dependencies
└── .env                        # Environment configuration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-ai-assistant.git
cd smart-ai-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Application

```bash
streamlit run main.py
```

### Interaction Modes

The application supports three personas:
- 🤝 Casual: Friendly, conversational tone
- 💼 Professional: Formal, concise responses
- 🔬 Technical: Precise, research-oriented communication

## Configuration

### Content Safety
Modify `src/safety_utils.py` to adjust:
- Inappropriate content patterns
- Risk level thresholds
- Filtering strategies

### Rate Limiting
In `src/safety_utils.py`, configure:
- Maximum retry attempts
- Base delay between retries

## Testing

Run unit tests:
```bash
python -m unittest discover tests
```

## Advanced Features

### Document Retrieval
- Supports CSV, JSON, and TXT formats
- Uses FAISS for semantic search
- Sentence transformer embeddings

### NLP Capabilities
- Text summarization
- Sentiment analysis
- Named Entity Recognition
- Question answering
- Code generation assistance

## Performance Optimization

- Disk-based caching
- Exponential backoff for API calls
- Efficient embedding and retrieval

## Ethical Considerations

- Content safety filtering
- Inappropriate content detection
- Configurable risk management

## Troubleshooting

- Ensure valid OpenAI API key
- Check internet connectivity
- Verify document format and content

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Contact

Rishab Rebala | rishabrebala1303@gmail.com

---

**Note**: This project is a demonstration of advanced AI assistant capabilities and should be used responsibly.