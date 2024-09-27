"""Using a strategy design pattern to create a client for each language model. This will allow us to easily add new language models in the future."""
from ollama import Client as Ollama
from openai import OpenAI
from typing import Optional
import prompts

class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def chat(self, messages, model) -> str:
        raise NotImplementedError

class OpenAIClient(LLMClient):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.client: OpenAI = OpenAI(api_key=api_key)

    def chat(self, messages, model="gpt-4o") -> str:
        self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]},
            ],
        )
        # TODO: return response from OpenAI

class OllamaClient(LLMClient):

    def __init__(self, api_key: Optional[str] = None, host: Optional[str] = None, model: str = "mistral-nemo"):
        super().__init__(api_key)
        self.client: Ollama = Ollama(host=host)
        ollama: Ollama = self.client
        modelfile=f'''
            FROM {model}
            SYSTEM {prompts.SYSTEM_PROMPT}
        '''
        ollama.create(model='resullme', modelfile=modelfile)
        

    def chat(self, messages: list[dict[str,str]], model="mistral-nemo") -> str:
        ollama: Ollama = self.client
        self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]},
            ],
        )