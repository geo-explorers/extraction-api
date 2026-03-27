from typing import Any, Dict
from src.config.settings import settings

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI


def _create_chat_model(
  model_name: str = None,
  temperature: float = None,
) -> BaseChatModel:
  return ChatGoogleGenerativeAI(
    model=model_name or settings.gemini_extraction_model,
    temperature=temperature if temperature is not None else settings.gemini_extraction_temperature,
    api_key=settings.gemini_api_key,
    max_retries=5,
  )

def build_chain(
  prompt: str,
  model_name: str = None,
  temperature: float = None,
) -> RunnableSequence[Dict[str, Any], str]:
  model = _create_chat_model(
    model_name=model_name,
    temperature=temperature,
  )
  prompt = ChatPromptTemplate.from_template(prompt)
  return prompt | model | StrOutputParser()