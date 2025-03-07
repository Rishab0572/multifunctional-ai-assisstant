import os
import openai
import logging
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMWrapper:
    def __init__(self, provider="openai", model_name=None):
        self.provider = provider.lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_model = model_name or os.getenv("HUGGINGFACE_MODEL", "facebook/bart-large-cnn")
        
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("Missing OpenAI API key. Set it in the .env file.")
        
        if self.provider == "huggingface":
            self.hf_pipeline = pipeline("text-generation", model=self.huggingface_model)

    def generate_response(self, prompt):
        """
        Generates AI response based on the selected provider.
        """
        try:
            if self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are an AI assistant."},
                              {"role": "user", "content": prompt}],
                    api_key=self.openai_api_key,
                    timeout=10  # Adding a timeout
                )
                return response["choices"][0]["message"]["content"]
            
            elif self.provider == "huggingface":
                response = self.hf_pipeline(prompt, max_length=150, num_return_sequences=1)
                return response[0]["generated_text"]
            
            else:
                raise ValueError("Unsupported provider. Choose 'openai' or 'huggingface'.")
        
        except Exception as e:
            logging.error(f"Error in LLM response generation: {str(e)}")
            return "Error generating response."

# Example Usage
if __name__ == "__main__":
    llm = LLMWrapper(provider="openai")
    print(llm.generate_response("What is AI?"))
