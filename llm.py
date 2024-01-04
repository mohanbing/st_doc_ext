# Copyright 2023 Aditya Mohan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
