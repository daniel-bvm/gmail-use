import os

os.environ.setdefault("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "http://localhost:65534/v1"))

from .agent import prompt
from . import models

__all__ = [
    "prompt",
    "models"
]