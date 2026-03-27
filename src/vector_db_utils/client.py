# src/vector_db_utils/client.py

"""
Низкоуровневый клиент для работы с Qdrant.
Поддерживает именованные векторы и все необходимые операции с коллекциями и точками.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter, 
    FieldCondition, 
    MatchValue, 
    Distance, 
    VectorParams, 
    PointStruct,
    MinShould
)
from .models import Chunk, VectorConfig
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
import asyncio
import time
import uuid
from functools import wraps
from contextlib import asynccontextmanager

import logging
logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, base_delay: float = 0.5):
    """Декоратор для повторных попыток при временных ошибках."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        jitter = delay * 0.1
                        await asyncio.sleep(delay + jitter)
                        logger.debug(f"Retry {attempt + 1}/{max_retries} for {func.__name__}")
                    else:
                        logger.error(f"All retries failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator


class QdrantClient:
    """
    Клиент для работы с Qdrant векторной БД.
    
    Особенности:
    - Ленивое подключение с автоматическим переподключением
    - Поддержка gRPC и HTTP режимов
    - Работа с именованными векторами
    - Retry при временных ошибках
    - Метрики и health check
    """
    
    def __init__(
            self,
            qdrant_host: str = "localhost", # Хост без протокола!
            api_key: Optional[str] = None,  # Статический API ключ
            grpc_port: int = 6334,
            grpc_enabled: bool = True,
            http_port: int = 6333,
            connection_timeout: float = 30.0,
            upsert_batch_size: int = 64,
            scroll_point_limit: int = 64
        ):
        self._client: Optional[AsyncQdrantClient] = None
        self._connection_lock = asyncio.Lock()
        
        # Параметры соединения 
        self._host = qdrant_host
        self._api_key = api_key
        self._connection_timeout = connection_timeout
        self._grpc_enabled = grpc_enabled
        self._grpc_port = grpc_port
        self._http_port = http_port

        # Настройки батчевых операций
        self._upsert_batch_size = upsert_batch_size
        self._scroll_point_limit = scroll_point_limit
        
        # Метрики для мониторинга
        self._total_requests = 0
        self._failed_requests = 0
        
        logger.debug("QdrantClient initialized")

    # ----------------------------------------------------------------------
    # Управление соединением
    # ----------------------------------------------------------------------

    async def get_client(self) -> AsyncQdrantClient:
        """
        Возвращает подключение к Qdrant.
        При отсутствии или сбое соединения создает новое.
        """
        if self._client is not None:
            try:
                await self._client.get_collections()
                return self._client
            except Exception as e:
                logger.warning(f"Existing connection is dead: {e}. Reconnecting...")
                self._client = None
        
        async with self._connection_lock:
            if self._client is not None:
                return self._client
            
            try:
                if self._grpc_enabled:
                    self._client = AsyncQdrantClient(
                        host=self._host,
                        api_key=self._api_key,
                        grpc_port=self._grpc_port,
                        prefer_grpc=True,
                        timeout=self._connection_timeout,
                        check_compatibility=False,
                    )
                    logger.info(f"Connected via gRPC: {self._host}:{self._grpc_port}")
                else:
                    self._client = AsyncQdrantClient(
                        url=f"http://{self._host}:{self._http_port}",
                        prefer_grpc=False,
                        timeout=self._connection_timeout,
                        check_compatibility=False,
                    )
                    logger.info(f"Connected via HTTP: {self._host}:{self._http_port}")
                
                await self._client.get_collections()
                logger.info("✅ Connection to Qdrant established")
                return self._client
                
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                self._client = None
                raise ConnectionError(f"Cannot connect to Qdrant: {e}")

    @asynccontextmanager
    async def operation_context(self, operation_name: str):
        """Контекстный менеджер для отслеживания операций."""
        start_time = time.time()
        self._total_requests += 1
        
        try:
            yield
            duration = time.time() - start_time
            logger.debug(f"Operation {operation_name} completed in {duration:.3f}s")
        except Exception as e:
            self._failed_requests += 1
            duration = time.time() - start_time
            logger.error(f"Operation {operation_name} failed after {duration:.3f}s: {e}")
            raise

    # ----------------------------------------------------------------------
    # Операции с коллекциями
    # ----------------------------------------------------------------------

    @retry_on_failure(max_retries=3)
    async def collection_exists(self, collection_name: str) -> bool:
        """Проверяет существование коллекции."""
        async with self.operation_context("collection_exists"):
            client = await self.get_client()
            collections = await client.get_collections()
            existing = [col.name for col in collections.collections]
            return collection_name in existing

    @retry_on_failure(max_retries=2)
    async def create_collection(
        self, 
        collection_name: str, 
        vectors_config: Dict[str, VectorConfig]
    ) -> bool:
        """
        Создает коллекцию с указанными именованными векторами.
        
        Args:
            collection_name: Имя коллекции
            vectors_config: Словарь {имя_вектора: VectorConfig}
        """
        async with self.operation_context("create_collection"):
            client = await self.get_client()
            
            qdrant_vectors_config = {}
            for vector_name, vector_config in vectors_config.items():
                qdrant_vectors_config[vector_name] = VectorParams(
                    size=vector_config.size,
                    distance=Distance[vector_config.distance.upper()]
                )
            
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_vectors_config,
                optimizers_config={
                    "default_segment_number": 2,
                    "indexing_threshold": 10000,
                }
            )
            
            vector_names = list(vectors_config.keys())
            logger.info(f"Collection created: {collection_name} with vectors: {vector_names}")
            return True

    @retry_on_failure(max_retries=2)
    async def delete_collection(self, collection_name: str) -> bool:
        """Удаляет коллекцию."""
        try:
            async with self.operation_context("delete_collection"):
                client = await self.get_client()
                await client.delete_collection(collection_name)
                logger.info(f"Collection deleted: {collection_name}")
                return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    async def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Возвращает информацию о коллекции."""
        try:
            async with self.operation_context("get_collection_info"):
                client = await self.get_client()
                info = await client.get_collection(collection_name)
                
                return {
                    "name": collection_name,
                    "points_count": info.points_count,
                    "segments_count": info.segments_count,
                    "status": info.status,
                    "config": info.config.model_dump()
                }
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None
        
    async def get_vectors_config(self, collection_name: str) -> Optional[Dict]:
        collection_info = await self.get_collection_info(collection_name)
        return collection_info.get("config", {}).get("params", {}).get("vectors", {})

    @retry_on_failure(max_retries=2)
    async def add_vectors_to_collection(
        self,
        collection_name: str,
        vectors_config: Dict[str, VectorParams]
    ) -> bool:
        """
        Добавляет новые векторы в существующую коллекцию.
        
        Qdrant позволяет добавлять новые именованные векторы к существующей коллекции.
        После добавления векторов их значения для существующих точек будут отсутствовать,
        их необходимо заполнить отдельно через обновление точек.
        
        Args:
            collection_name: Имя коллекции
            vectors_config: Словарь {имя_вектора: VectorParams} с новыми векторами
            
        Returns:
            bool: True если успешно, False при ошибке
        """
        async with self.operation_context("add_vectors_to_collection"):
            client = await self.get_client()
            
            try:
                await client.update_collection(
                    collection_name=collection_name,
                    vectors_config=vectors_config
                )
                logger.info(f"Added vectors to {collection_name}: {list(vectors_config.keys())}")
                return True
            except Exception as e:
                logger.error(f"Failed to add vectors to {collection_name}: {e}")
                return False

    # ----------------------------------------------------------------------
    # Операции с точками
    # ----------------------------------------------------------------------

    async def _simple_upsert(self, collection_name: str, points: List[Dict[str, Any]]) -> int:
        """
        Внутренний метод для простой вставки точек без проверок и миграций.
        
        Args:
            collection_name: Имя коллекции
            points: Список точек в формате:
                   [{
                       "id": str,
                       "vector": {"rag": [0.1, ...], "classifier": [0.2, ...]},
                       "payload": {...}
                   }]
        
        Returns:
            int: Количество успешно вставленных точек
        """
        if not points:
            return 0
        
        client = await self.get_client()
        
        point_structs = []
        for point in points:
            point_structs.append(PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point.get("payload", {})
            ))
        
        total_inserted = 0
        
        for i in range(0, len(point_structs), self._upsert_batch_size):
            batch = point_structs[i: i + self._upsert_batch_size]
            try:
                await client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
                total_inserted += len(batch)
            except Exception as e:
                logger.error(f"Failed to upsert batch to {collection_name}: {e}")
        
        return total_inserted

    async def _prepare_points_for_upsert(self, points: List[Dict[str, Any]]) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Подготавливает точки к вставке: фильтрует валидные векторы и собирает информацию о размерах.
        
        Returns:
            Tuple[List[Dict], Dict[str, int]]: (отфильтрованные точки, словарь {имя_вектора: размер})
        """
        valid_points = []
        vector_sizes = {}
        
        for point in points:
            valid_vectors = {}
            for vec_name, vec_value in point["vector"].items():
                if vec_value is not None:
                    valid_vectors[vec_name] = vec_value
                    
                    if vec_name not in vector_sizes:
                        vector_sizes[vec_name] = len(vec_value)
                    elif len(vec_value) != vector_sizes[vec_name]:
                        logger.error(f"Vector '{vec_name}' has inconsistent sizes across points")
                        return [], {}
            
            if valid_vectors:
                valid_points.append({
                    "id": point["id"],
                    "vector": valid_vectors,
                    "payload": point.get("payload", {})
                })
        
        return valid_points, vector_sizes

    async def _create_collection_from_points(self, collection_name: str, points: List[Dict], vector_sizes: Dict[str, int]) -> bool:
        """Создает новую коллекцию на основе размеров векторов из точек."""
        client = await self.get_client()
        
        vectors_config = {}
        for vector_name, size in vector_sizes.items():
            vectors_config[vector_name] = VectorParams(
                size=size,
                distance=Distance.COSINE
            )
        
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
        logger.info(f"Created new collection {collection_name} with vectors: {list(vector_sizes.keys())}")
        return True

    async def _recreate_collection_with_new_vectors(
        self,
        collection_name: str,
        existing_points: List[Dict],
        new_points: List[Dict],
        current_vectors_config: Dict,
        new_vectors: Set[str],
        vector_sizes_in_points: Dict[str, int]
    ) -> int:
        """
        Пересоздает коллекцию с расширенным набором векторов.
        
        Используется когда нужно добавить новые векторы, но при этом необходимо
        пересоздать коллекцию (например, если точки уже содержат значения для новых векторов).
        
        Returns:
            int: Количество новых точек, успешно добавленных в коллекцию
        """
        client = await self.get_client()
        
        # Создаем конфигурацию для новой коллекции
        new_vectors_config = {}
        
        # Добавляем существующие векторы
        for vec_name, vec_config in current_vectors_config.items():
            new_vectors_config[vec_name] = VectorParams(
                size=vec_config["size"],
                distance=Distance[vec_config["distance"].upper()]
            )
        
        # Добавляем новые векторы
        for vec_name in new_vectors:
            new_vectors_config[vec_name] = VectorParams(
                size=vector_sizes_in_points[vec_name],
                distance=Distance.COSINE
            )
        
        # Создаем временную коллекцию
        temp_collection_name = f"{collection_name}_temp_{uuid.uuid4().hex[:8]}"
        
        try:
            await client.create_collection(
                collection_name=temp_collection_name,
                vectors_config=new_vectors_config
            )
            logger.info(f"Created temporary collection: {temp_collection_name}")
            
            # Переносим существующие точки
            if existing_points:
                migrated = await self._simple_upsert(temp_collection_name, existing_points)
                if migrated < len(existing_points):
                    logger.warning(f"Only {migrated}/{len(existing_points)} existing points migrated")
            
            # Добавляем новые точки
            new_points_inserted = 0
            if new_points:
                new_points_inserted = await self._simple_upsert(temp_collection_name, new_points)
                if new_points_inserted < len(new_points):
                    logger.warning(f"Only {new_points_inserted}/{len(new_points)} new points inserted")
            
            # Заменяем старую коллекцию на временную
            await client.delete_collection(collection_name)
            logger.info(f"Deleted old collection: {collection_name}")
            
            # Переименовываем временную коллекцию
            # Qdrant не поддерживает переименование, поэтому создаем новую с правильным именем
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=new_vectors_config
            )
            
            # Переносим все точки из временной в новую
            temp_points = await self._scroll_all_points(temp_collection_name)
            if temp_points:
                final_inserted = await self._simple_upsert(collection_name, temp_points)
                if final_inserted < len(temp_points):
                    logger.warning(f"Only {final_inserted}/{len(temp_points)} points restored")
            
            logger.info(f"Collection {collection_name} recreated with vectors: {list(new_vectors_config.keys())}")
            return new_points_inserted
            
        finally:
            try:
                await client.delete_collection(temp_collection_name)
                logger.info(f"Cleaned up temporary collection: {temp_collection_name}")
            except Exception as e:
                logger.error(f"Failed to delete temporary collection {temp_collection_name}: {e}")

    @retry_on_failure(max_retries=2)
    async def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]) -> int:
        """
        Сохраняет точки с именованными векторами.
        
        Алгоритм работы:
        1. Проверяет существование коллекции
        2. Если коллекции нет - создает её и вставляет точки
        3. Если коллекция есть:
           - Проверяет конфликты размерности существующих векторов
           - Определяет новые векторы в точках
           - Если новых векторов нет - простая вставка
           - Если есть новые векторы - пересоздает коллекцию с расширенной схемой
        
        Args:
            collection_name: Имя коллекции
            points: Список точек в формате:
                   [{
                       "id": str,
                       "vector": {"rag": [0.1, ...], "classifier": [0.2, ...]},
                       "payload": {...}
                   }]
        
        Returns:
            int: Количество успешно добавленных точек
        """
        if not points:
            logger.error("No points to upsert")
            return 0
        
        async with self.operation_context("upsert_points"):
            
            # Шаг 1: Подготавливаем точки
            valid_points, vector_sizes = await self._prepare_points_for_upsert(points)
            
            if not valid_points:
                logger.warning("No valid points to upsert")
                return 0
            
            # Шаг 2: Проверяем существование коллекции
            collection_exists = await self.collection_exists(collection_name)
            
            if not collection_exists:
                await self._create_collection_from_points(collection_name, valid_points, vector_sizes)
                return await self._simple_upsert(collection_name, valid_points)
            
            # Шаг 3: Коллекция существует - проверяем конфигурацию
            info = await self.get_collection_info(collection_name)
            if not info:
                logger.error(f"Cannot get info for collection {collection_name}")
                return 0
            
            current_vectors_config: Dict = info.get("config", {}).get("params", {}).get("vectors", {})
            existing_vector_names = set(current_vectors_config.keys())
            vectors_in_points = set(vector_sizes.keys())
            
            # Шаг 4: Проверяем конфликты размерности
            for vec_name in vectors_in_points.intersection(existing_vector_names):
                expected_size = current_vectors_config[vec_name]["size"]
                actual_size = vector_sizes[vec_name]
                if actual_size != expected_size:
                    logger.error(
                        f"Vector '{vec_name}' size mismatch: "
                        f"collection expects {expected_size}, point has {actual_size}"
                    )
                    return 0
            
            # Шаг 5: Определяем новые векторы
            new_vectors = vectors_in_points - existing_vector_names
            
            if not new_vectors:
                return await self._simple_upsert(collection_name, valid_points)
            
            # Шаг 6: Есть новые векторы - нужно пересоздать коллекцию
            # Получаем существующие точки (они понадобятся при пересоздании)
            existing_points = await self._scroll_all_points(collection_name)
            logger.info(f"Found {len(existing_points)} existing points in collection")
            
            logger.info(f"New vectors detected: {new_vectors}. Recreating collection...")
            return await self._recreate_collection_with_new_vectors(
                collection_name,
                existing_points,
                valid_points,
                current_vectors_config,
                new_vectors,
                vector_sizes
            )

    @retry_on_failure(max_retries=3)
    async def upsert_chunks(
        self, 
        collection_name: str, 
        chunks: List[Chunk], 
        document_name: str, 
        tenant_id: str
    ) -> int:
        """
        Сохраняет чанки документа в коллекцию.
        
        Args:
            collection_name: Имя коллекции
            chunks: Список чанков со словарем vectors
            document_name: Имя документа
            tenant_id: ID тенанта
        
        Returns:
            int: Количество успешно сохраненных чанков
        """
        if not chunks:
            logger.error("No chunks to upsert")
            return 0
        
        async with self.operation_context("upsert_chunks"):
            
            points = []
            for i, chunk in enumerate(chunks):
                if not chunk.vectors:
                    logger.error(f"Chunk {i} has no vectors")
                    continue
                
                payload = {
                    "document_name": document_name,
                    "chunk_index": i,
                    "content": chunk.content,
                    "tenant": tenant_id
                }
                
                if chunk.title:
                    payload["chunk_title"] = chunk.title
                if chunk.topics:
                    payload["tags"] = chunk.topics
                
                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": chunk.vectors,
                    "payload": payload
                })
            
            if not points:
                logger.error("No valid points created")
                return 0
            
            return await self.upsert_points(collection_name, points)

    @retry_on_failure(max_retries=3)
    async def point_exists(self, collection_name: str, point_id: str) -> bool:
        """
        Проверяет, существует ли точка с указанным ID.
        
        Args:
            collection_name: Имя коллекции
            point_id: ID точки
        """
        async with self.operation_context("point_exists"):
            client = await self.get_client()
            try:
                points = await client.retrieve(
                    collection_name=collection_name,
                    ids=[point_id],
                    with_vectors=False,
                    with_payload=False
                )
                return len(points) > 0
            except Exception:
                return False

    async def search_by_tags_or(
    self,
    collection_name: str,
    query_vector: List[float],
    vector_name: str,
    tenant_id: str,
    tags: Optional[List[str]] = None,
    limit: int = 10,
    score_threshold: float = 0.5
) -> List[Dict[str, Any]]:
        """
        Поиск точек с OR-фильтрацией по тегам, используя указанный вектор.
        
        Args:
            collection_name: Имя коллекции
            query_vector: Вектор запроса
            vector_name: Имя вектора для поиска
            tenant_id: ID тенанта
            tags: Список тегов для OR-фильтрации
            limit: Максимальное количество результатов
            score_threshold: Порог релевантности

        Returns:
            List[Dict[str, Any]]: Список результатов с полями id, score, payload
        """
        try:
            async with self.operation_context("search_by_tags_or"):
                client = await self.get_client()
                
                # Формируем условия фильтрации
                must_conditions = [
                    FieldCondition(key="tenant", match=MatchValue(value=tenant_id))
                ]
                
                if tags:
                    should_conditions = [
                        FieldCondition(key="tags", match=MatchValue(value=tag))
                        for tag in tags
                    ]
                    search_filter = Filter(
                        must=must_conditions,
                        should=should_conditions,
                        min_should=MinShould(conditions=should_conditions, min_count=1)
                    )
                else:
                    search_filter = Filter(must=must_conditions)
                
                # Используем query_points
                response = await client.query_points(
                    collection_name=collection_name,
                    query=query_vector,             # вектор запроса
                    using=vector_name,              # имя вектора для поиска
                    query_filter=search_filter,     # фильтр
                    limit=limit,                    # лимит результатов
                    score_threshold=score_threshold,# порог релевантности
                    with_payload=True,              # возвращаем payload
                    with_vectors=False              # не возвращаем векторы (экономия памяти)
                )
                
                # response.points содержит список результатов
                results = response.points
                
                # Преобразуем в удобный формат
                return [{
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                } for hit in results]
                    
        except Exception as e:
            logger.warning(f"Search failed in {collection_name} with vector '{vector_name}': {e}")
            return []

    async def scroll_points(
        self,
        collection_name: str,
        filter: Optional[Filter] = None,
        offset = None
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Получает точки по фильтру с пагинацией.
        
        Returns:
            tuple: (список точек, следующий offset для продолжения)
        """
        try:
            async with self.operation_context("scroll_points"):
                client = await self.get_client()
                
                points, next_offset = await client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter,
                    limit=self._scroll_point_limit,
                    offset=offset,
                    with_vectors=True, 
                    with_payload=True,
                )
                
                result = [{
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector,
                    "score": 1.0
                } for point in points]
                
                return result, next_offset
                
        except Exception as e:
            logger.error(f"Error scrolling in {collection_name}: {e}")
            return [], None

    async def _scroll_all_points(self, collection_name: str) -> List[Dict[str, Any]]:
        """Внутренний метод для получения всех точек коллекции."""
        all_points = []
        offset = None
        
        while True:
            batch, offset = await self.scroll_points(
                collection_name=collection_name,
                filter=None,
                offset=offset
            )
            if not batch:
                break
            all_points.extend(batch)
            if offset is None:
                break
        
        return all_points

    async def delete_points_by_filter(self, collection_name: str, filter: Filter) -> bool:
        """Удаляет точки по фильтру."""
        async with self.operation_context("delete_points_by_filter"):
            client = await self.get_client()
            
            await client.delete(
                collection_name=collection_name,
                points_selector=filter,
                wait=True
            )
            
            logger.info(f"Points deleted from {collection_name}")
            return True

    @retry_on_failure(max_retries=2)
    async def remove_vector_completely(self, collection_name: str, vector_name: str) -> bool:
        """
        Полностью удаляет вектор из коллекции:
        1. Удаляет вектор из всех точек
        2. Удаляет вектор из конфигурации коллекции
        
        Args:
            collection_name: Имя коллекции
            vector_name: Имя удаляемого вектора
            
        Returns:
            bool: True если успешно, False при ошибке
        """
        async with self.operation_context("remove_vector_completely"):
            client = await self.get_client()
            
            try:
                # Получаем все точки
                all_points = await self._scroll_all_points(collection_name)
                logger.info(f"Found {len(all_points)} total points in collection")
                
                # Получаем текущую конфигурацию векторов
                info = await self.get_collection_info(collection_name)
                if not info:
                    logger.warning(f"Collection {collection_name} not found")
                    return False
                
                current_vectors_config: Dict = info.get("config", {}).get("params", {}).get("vectors", {})
                
                if vector_name not in current_vectors_config:
                    logger.debug(f"Vector '{vector_name}' not in collection config")
                    return True
                
                # Создаем новую конфигурацию без удаляемого вектора
                new_vectors_config = {}
                for vec_name, vec_config in current_vectors_config.items():
                    if vec_name != vector_name:
                        new_vectors_config[vec_name] = VectorParams(
                            size=vec_config["size"],
                            distance=Distance[vec_config["distance"].upper()]
                        )
                
                if not new_vectors_config:
                    logger.error("Cannot remove last vector from collection")
                    return False
                
                # Создаем временную коллекцию
                temp_collection_name = f"{collection_name}_temp_{uuid.uuid4().hex[:8]}"
                
                try:
                    # Создаем временную коллекцию без удаляемого вектора
                    await client.create_collection(
                        collection_name=temp_collection_name,
                        vectors_config=new_vectors_config,
                        optimizers_config={
                            "default_segment_number": 2,
                            "indexing_threshold": 10000,
                        }
                    )
                    
                    # Убираем удаленный вектор из точек
                    cleaned_points = []
                    for point in all_points:
                        if vector_name in point.get("vector", {}):
                            updated_vectors = point["vector"].copy()
                            del updated_vectors[vector_name]
                            point["vector"] = updated_vectors
                        cleaned_points.append(point)
                    
                    # Переносим точки во временную коллекцию
                    if cleaned_points:
                        inserted = await self._simple_upsert(temp_collection_name, cleaned_points)
                        if inserted < len(cleaned_points):
                            logger.warning(f"Only {inserted}/{len(cleaned_points)} points restored")
                            return False
                    
                    # Удаляем старую коллекцию
                    await client.delete_collection(collection_name)
                    logger.info(f"Deleted old collection: {collection_name}")
                    
                    # Создаем новую коллекцию с правильным именем
                    await client.create_collection(
                        collection_name=collection_name,
                        vectors_config=new_vectors_config,
                        optimizers_config={
                            "default_segment_number": 2,
                            "indexing_threshold": 10000,
                        }
                    )
                    
                    # Переносим точки из временной в новую
                    temp_points = await self._scroll_all_points(temp_collection_name)
                    if temp_points:
                        final_inserted = await self._simple_upsert(collection_name, temp_points)
                        if final_inserted < len(temp_points):
                            logger.warning(f"Only {final_inserted}/{len(temp_points)} points restored")
                    
                    logger.info(f"Completely removed vector '{vector_name}' from collection {collection_name}")
                    return True
                    
                finally:
                    try:
                        await client.delete_collection(temp_collection_name)
                        logger.info(f"Cleaned up temporary collection: {temp_collection_name}")
                    except Exception as e:
                        logger.error(f"Failed to delete temporary collection {temp_collection_name}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to completely remove vector '{vector_name}': {e}")
                return False

    # ----------------------------------------------------------------------
    # Вспомогательные методы для работы с фильтрами
    # ----------------------------------------------------------------------

    def build_tenant_filter(self, tenant_id: str) -> Filter:
        """Фильтр только по tenant."""
        return Filter(
            must=[FieldCondition(key="tenant", match=MatchValue(value=tenant_id))]
        )

    def build_document_filter(self, tenant_id: str, document_name: str) -> Filter:
        """Фильтр для поиска документов по имени."""
        return Filter(
            must=[
                FieldCondition(key="tenant", match=MatchValue(value=tenant_id)),
                FieldCondition(key="document_name", match=MatchValue(value=document_name))
            ]
        )

    def build_tag_filter(self, tenant_id: str, tag: str) -> Filter:
        """Фильтр для поиска по одному тегу."""
        return Filter(
            must=[
                FieldCondition(key="tenant", match=MatchValue(value=tenant_id)),
                FieldCondition(key="tags", match=MatchValue(value=tag))
            ]
        )

    def build_tags_and_filter(self, tenant_id: str, tags: List[str]) -> Filter:
        """Фильтр для поиска по нескольким тегам (AND)."""
        must_conditions = [
            FieldCondition(key="tenant", match=MatchValue(value=tenant_id))
        ]
        for tag in tags:
            must_conditions.append(
                FieldCondition(key="tags", match=MatchValue(value=tag))
            )
        return Filter(must=must_conditions)
    
    # ----------------------------------------------------------------------
    # Получение информации о коллекциях и документах
    # ----------------------------------------------------------------------

    async def get_unique_field_values(
        self, 
        collection_name: str, 
        field_name: str,
        scroll_limit: Optional[int] = None
    ) -> Tuple[set, int]:
        """
        Возвращает уникальные значения поля и их общее количество.
        
        Args:
            collection_name: Имя коллекции
            field_name: Имя поля в payload
            scroll_limit: Просматривать не более данного количества точек.
                Если None - возвращаются все найденные значения.
        
        Returns:
            tuple: (множество уникальных значений, общее количество уникальных значений)
        """
        values = set()
        offset = None
        points_processed = 0
        
        while True:
            batch, offset = await self.scroll_points(
                collection_name=collection_name,
                filter=None,
                offset=offset
            )
            
            for point in batch:
                if field_name in point["payload"]:
                    values.add(point["payload"][field_name])
                points_processed += 1
                
                if scroll_limit and points_processed >= scroll_limit:
                    return values, len(values)
            
            if offset is None:
                break
        
        return values, len(values)

    # ----------------------------------------------------------------------
    # Health check и метрики
    # ----------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Проверка здоровья соединения."""
        try:
            client = await self.get_client()
            await client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Статистика работы клиента."""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "error_rate": self._failed_requests / max(self._total_requests, 1),
            "connected": self._client is not None,
            "grpc_enabled": self._grpc_enabled,
        }

    async def close(self):
        """Закрытие соединения."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("QdrantClient connection closed")