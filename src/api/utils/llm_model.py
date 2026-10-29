
from typing import Any, Dict
from src.config.overrides import llm
from src.infrastructure import gemini_ladder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI


def _create_chat_model(
  model_name: str = None,
  temperature: float = None,
) -> Runnable:
  # Resolved here, inside the caller's override scope, as before the ladder.
  model = model_name or llm.get("gemini_extraction_model")
  temperature = temperature if temperature is not None else llm.get("gemini_extraction_temperature")

  def invoke(prompt_value):
    # One chat model per key: the ladder picks the key (see gemini_ladder).
    return gemini_ladder.run(
      model,
      lambda key: ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=key,
        max_retries=3,
      ).invoke(prompt_value),
    )

  return RunnableLambda(invoke)

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
