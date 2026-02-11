import os
from abc import ABC, abstractmethod
import google.generativeai as genai
import openai
import ollama
import requests  

class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @classmethod
    @abstractmethod
    def provider_instance(cls, model_name: str) -> bool:
        """Check if this provider can handle the given model name."""
        pass

    @property
    @abstractmethod
    def env_var_name(self) -> str:
        """The name of the environment variable required for the API key."""
        pass

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """Generates content based on the prompt."""
        pass


class GeminiProvider(ModelProvider):
    """Provider for Google's Gemini models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        # In GeminiProvider the order is first config_api_key for backward compatibility
        self.api_key = config_api_key or os.getenv(self.env_var_name)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        """For backward compatibility, Gemini is the default provider."""
        return True

    @property
    def env_var_name(self) -> str:
        return "GEMINI_API_KEY"
        
    def generate_content(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("gpt") or model_name.startswith("o1")

    @property
    def env_var_name(self) -> str:
        return "OPENAI_API_KEY"
        
    def generate_content(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class OllamaProvider(ModelProvider):
    """Provider for Ollama models."""

    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        self.model_name = model_name.lstrip("ollama/")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = ollama.Client(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            headers=headers
        )

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("ollama/")

    @property
    def env_var_name(self) -> str:
        return "OLLAMA_API_KEY"

    def generate_content(self, prompt: str) -> str:
        response = self.client.chat(
            self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content


class HuggingFaceProvider(ModelProvider):
    """
    Provider for Hugging Face Inference API (serverless).
    Use model names like: hf/meta-llama/Meta-Llama-3-8B-Instruct
    """
    def __init__(self, config_api_key: str, model_name: str = "hf/Qwen/Qwen3-Coder-Next"):
        
        # Accept either HF_TOKEN or HUGGINGFACEHUB_API_TOKEN (common on HF setups)
        self.api_key = (
            config_api_key
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if not self.api_key:
            raise ValueError(
                f"Missing HF token for {model_name}. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
            )

        self.model_id = model_name.split("/", 1)[1]  # strip "hf/"
        base = os.getenv("HF_INFERENCE_BASE_URL", "https://router.huggingface.co/v1/chat/completions").rstrip("/")
        print("HF intialized")
        self.api_url = f"{base}"
        self.session = requests.Session()

        # Basic defaults (you can tune these)
        # self.max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "1024"))
        # self.temperature = float(os.getenv("HF_TEMPERATURE", "0.2"))

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("hf/") or model_name.startswith("huggingface/")

    @property
    def env_var_name(self) -> str:
        # Primary env var we recommend (even though we also accept HUGGINGFACEHUB_API_TOKEN)
        return "HF_TOKEN"

    def generate_content(self, prompt: str) -> str:
        print(f"Sending request to HF Inference API for model {self.model_id}...")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", 
                          "content": prompt}],
        }

        r = self.session.post(self.api_url, headers=headers, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()

        # Common HF formats:
        # - [{"generated_text": "..."}]
        # - {"generated_text": "..."}
        # - {"error": "...", "estimated_time": ...}
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF Inference error: {data.get('error')} (details: {data})")

        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]

        # Fallback: return raw JSON if HF changed format
        return data['choices'][0]['message']['content']
