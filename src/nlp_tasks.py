import torch
import logging
from transformers import (
    pipeline, CodeGenForCausalLM, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

device = get_device()

def load_summarization_model():
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        return pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if device != "cpu" else -1)
    except Exception as e:
        logging.error(f"Error loading summarization model: {str(e)}")
        return None

def summarize_text(text, summarizer, max_length=150, min_length=50):
    if not summarizer:
        return "Summarization model not loaded."
    
    try:
        input_text = " ".join(sent_tokenize(text)[:5])
        summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        return summary[0]['summary_text']
    except Exception as e:
        logging.error(f"Error summarizing text: {str(e)}")
        return "Error summarizing text."

def load_sentiment_model():
    try:
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        return pipeline("sentiment-analysis", model=model_name, device=0 if device != "cpu" else -1)
    except Exception as e:
        logging.error(f"Error loading sentiment model: {str(e)}")
        return None

def analyze_sentiment(text, classifier):
    if not classifier:
        return "Sentiment model not loaded."
    
    try:
        result = classifier(text)
        return result[0]['label']
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {str(e)}")
        return "Error analyzing sentiment."

def load_ner_model():
    try:
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        return pipeline("ner", model=model_name, tokenizer=model_name, device=0 if device != "cpu" else -1)
    except Exception as e:
        logging.error(f"Error loading NER model: {str(e)}")
        return None

def extract_entities(text, ner_pipeline):
    if not ner_pipeline:
        return "NER model not loaded."
    
    try:
        return [(entity['word'], entity['entity']) for entity in ner_pipeline(text)]
    except Exception as e:
        logging.error(f"Error extracting entities: {str(e)}")
        return "Error extracting entities."

def load_question_answering_model():
    try:
        model_name = "deepset/roberta-base-squad2"
        return pipeline("question-answering", model=model_name, tokenizer=model_name, device=0 if device != "cpu" else -1)
    except Exception as e:
        logging.error(f"Error loading QA model: {str(e)}")
        return None

def answer_question(question, context, qa_pipeline):
    if not qa_pipeline:
        return "QA model not loaded."
    
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        logging.error(f"Error in QA: {str(e)}")
        return "Error answering question."

def load_code_generation_model():
    try:
        model_name = "Salesforce/codegen-350M-multi"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = CodeGenForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32)
        model.to(device)
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device != "cpu" else -1)
    except Exception as e:
        logging.error(f"Error loading code generation model: {str(e)}")
        return None

def generate_code(prompt, code_gen_pipeline, max_length=50):
    if not code_gen_pipeline:
        return "Code generation model not loaded."
    
    try:
        code = code_gen_pipeline(prompt, max_length=max_length, do_sample=True, truncation=True)
        return code[0]['generated_text']
    except Exception as e:
        logging.error(f"Error generating code: {str(e)}")
        return "Error generating code."

# Load models
summarizer = load_summarization_model()
sentiment_classifier = load_sentiment_model()
ner_pipeline = load_ner_model()
qa_pipeline = load_question_answering_model()
code_gen_pipeline = load_code_generation_model()
