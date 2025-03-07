import sys
import os
import unittest
import faiss
from evaluation import evaluate_retrieval

# Add the `src` folder to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retrieval import retrieve_similar_documents, process_file, clean_text  # Import additional function for testing

class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup FAISS index for testing."""
        file_path = "data/sample.txt"  # Ensure this exists
        cls.data, cls.texts, cls.embeddings, cls.index = process_file(file_path)

    def test_retrieval(self):
        """Test if relevant documents are retrieved."""
        query = "What are lung cancer symptoms?"
        retrieved_docs = retrieve_similar_documents(query, self.index, self.texts, top_k=3)
        self.assertGreater(len(retrieved_docs), 0, "No documents retrieved.")

    def test_empty_query(self):
        """Test retrieval function with an empty query."""
        query = ""
        retrieved_docs = retrieve_similar_documents(query, self.index, self.texts, top_k=3)
        self.assertEqual(len(retrieved_docs), 0, "Empty query should return no results.")

    def test_non_existent_query(self):
        """Test retrieval with a query that has no matches."""
        query = "This is a completely unrelated topic"
        retrieved_docs = retrieve_similar_documents(query, self.index, self.texts, top_k=3)
        self.assertEqual(len(retrieved_docs), 0, "Non-matching query should return no results.")

    def test_faiss_index_size(self):
        """Ensure FAISS index contains the expected number of embeddings."""
        self.assertEqual(self.index.ntotal, len(self.texts), "FAISS index size does not match number of texts.")

    def test_text_preprocessing(self):
        """Test text cleaning and normalization."""
        raw_text = "Lung cancer!! is A deadly DISEASE."
        expected_cleaned = "lung cancer deadly disease"
        self.assertEqual(clean_text(raw_text), expected_cleaned, "Text preprocessing failed.")

class TestRetrievalEvaluation(unittest.TestCase):
    def test_retrieval(self):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        relevant_docs = ["doc1", "doc3"]
        scores = evaluate_retrieval(retrieved_docs, relevant_docs, k=3)
        print("Retrieval Scores:", scores)

if __name__ == "__main__":
    unittest.main()
