# src/vector_db_utils/__init__.py

from .client import VBDClient
from .tag_normalizer import normalize_tag, normalize_tag_list, normalize_tag_set
from .models import Chunk, VectorConfig

# Это то, что будет доступно при "from vector_db_utils import *"
__all__ = [
    "VBDClient",
    "normalize_tag",
    "normalize_tag_list",
    "normalize_tag_set",
    "Chunk",
    "VectorConfig"
]