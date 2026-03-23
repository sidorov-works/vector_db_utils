# shared/vector_db/models.py

"""
Pydantic модели для работы с векторной БД.
Обновлено для поддержки именованных векторов (несколько моделей энкодеров).
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Chunk(BaseModel):
    """
    Чанк документа для индексации.
    
    Атрибуты:
        content: Текст чанка
        vectors: Словарь именованных векторов {имя_энкодера: список float}
        title: Заголовок чанка (опционально)
        topics: Список тем/тегов от LLM
        original_index: Оригинальный индекс для многочастевых документов
    """
    content: str
    vectors: Dict[str, List[float]]  # ключ - имя энкодера, значение - вектор
    title: Optional[str] = None
    topics: Optional[List[str]] = None
    original_index: Optional[int] = None


class VectorConfig(BaseModel):
    """
    Конфигурация вектора для создания коллекции в Qdrant.
    
    Qdrant поддерживает несколько именованных векторов в одной коллекции,
    каждый со своей размерностью и метрикой.
    """
    size: int
    distance: str = "Cosine"  # Qdrant принимает "Cosine", "Euclid", "Dot"