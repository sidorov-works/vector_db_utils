# src/vector_db_utils/tag_normalizer.py

"""
Нормализация тегов для единообразного хранения и поиска в векторной БД.

Теги приводятся к единому формату:
- нижний регистр
- пробелы заменяются на подчеркивания
- удаляются лишние пробелы и множественные подчеркивания

Примеры:
    "описание устройства" → "описание_устройства"
    "  Техническая Поддержка  " → "техническая_поддержка"
    "передняя панель" → "передняя_панель"
    "гарантия   возврат" → "гарантия_возврат"
"""

from typing import List, Optional


def normalize_tag(tag: Optional[str]) -> Optional[str]:
    """
    Нормализует один тег.
    
    Args:
        tag: Исходный тег (может быть None)
        
    Returns:
        Optional[str]: Нормализованный тег или None если входной был None
    """
    if not tag:
        return tag
    
    # Приводим к нижнему регистру и убираем пробелы по краям
    normalized = tag.lower().strip()
    
    # Заменяем пробелы на подчеркивания
    normalized = normalized.replace(' ', '_')
    
    # Убираем множественные подчеркивания (от двойных пробелов)
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    
    # Убираем подчеркивания в начале и конце
    normalized = normalized.strip('_')
    
    return normalized


def normalize_tag_list(tags: List[str]) -> List[str]:
    """
    Нормализует список тегов.
    
    Args:
        tags: Список исходных тегов
        
    Returns:
        List[str]: Список нормализованных тегов (пустые и None пропускаются)
    """
    if not tags:
        return []
    
    normalized = []
    for tag in tags:
        norm_tag = normalize_tag(tag)
        if norm_tag:  # пропускаем пустые после нормализации
            normalized.append(norm_tag)
    
    return normalized


def normalize_tag_set(tags: List[str]) -> List[str]:
    """
    Нормализует список тегов и удаляет дубликаты.
    Сохраняет порядок первого вхождения.
    
    Args:
        tags: Список исходных тегов
        
    Returns:
        List[str]: Список уникальных нормализованных тегов
    """
    if not tags:
        return []
    
    seen = set()
    unique_normalized = []
    
    for tag in tags:
        norm_tag = normalize_tag(tag)
        if norm_tag and norm_tag not in seen:
            seen.add(norm_tag)
            unique_normalized.append(norm_tag)
    
    return unique_normalized