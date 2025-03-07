import torch
from transformers import pipeline
import unittest
from evaluation import evaluate_summarization, evaluate_sentiment, evaluate_ner, evaluate_question_answering

class TestNLPTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize pipelines with explicit models and correct max_length settings."""
        cls.device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else -1

        cls.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=cls.device)
        cls.classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=cls.device)
        cls.ner_pipeline = pipeline("ner", model="bert-base-cased", device=cls.device)  # Fixed NER model
        cls.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=cls.device)

    def test_summarization(self):
        text = "This is a long article about AI and healthcare."
        max_len = min(len(text.split()), 50)  # Fix: dynamic max_length
        result = self.summarizer(text, max_length=max_len, min_length=5, do_sample=False)
        self.assertIsInstance(result, list)

    def test_sentiment_analysis(self):
        result = self.classifier("I love this product!")
        self.assertIsInstance(result, list)

    def test_ner(self):
        result = self.ner_pipeline("Barack Obama was born in Hawaii.")
        self.assertIsInstance(result, list)

    def test_question_answering(self):
        context = "The Eiffel Tower is located in Paris."
        question = "Where is the Eiffel Tower?"
        result = self.qa_pipeline(question=question, context=context)
        self.assertIn("answer", result)

class TestNLPEvaluation(unittest.TestCase):
    def test_summarization(self):
        generated_summary = "AI is transforming the world."
        reference_summary = "Artificial Intelligence is changing industries."
        scores = evaluate_summarization(generated_summary, reference_summary)
        print("Summarization Scores:", scores)

    def test_sentiment_analysis(self):
        predictions = [1, 0, 1, 1]  # Example model output
        labels = [1, 0, 0, 1]  # Ground truth
        scores = evaluate_sentiment(predictions, labels)
        print("Sentiment Analysis Scores:", scores)

    def test_ner(self):
        predictions = ["O", "B-PER", "I-PER", "O", "B-LOC"]
        labels = ["O", "B-PER", "I-PER", "O", "B-LOC"]
        scores = evaluate_ner(predictions, labels)
        print("NER Scores:", scores)

    def test_question_answering(self):
        predicted_answers = ["Paris", "Machine Learning"]
        actual_answers = ["Paris", "AI and Machine Learning"]
        scores = evaluate_question_answering(predicted_answers, actual_answers)
        print("QA Scores:", scores)


if __name__ == "__main__":
    unittest.main()
