"""
LLM Handler Code
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain


class LLM:
    """Handles communication with OpenAI LLMs"""

    def __init__(self, temperature: float, openai_api_key: str) -> None:
        self.llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)

    def analyze_text(self, text: str, schema: dict) -> dict:
        """Analyze text according to schema

        Args:
            text (str): OCR Output to be analyzed
            schema (dict): Schema to be processed

        Returns:
            dict: LLM Response
        """
        chain = create_extraction_chain(schema, self.llm)
        return chain.run(text)
