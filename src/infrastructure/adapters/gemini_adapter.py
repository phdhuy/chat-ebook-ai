import google.generativeai as genai

from src.application.interfaces import ILLMService


class GeminiAdapter(ILLMService):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def generate(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text or "No response generated."
