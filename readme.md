# vector-db-utils

Асинхронный Python клиент для работы с Qdrant векторной базой данных с поддержкой именованных векторов.

## Особенности

- **Именованные векторы** — поддержка нескольких векторов в одной точке
- **Асинхронный** — полностью `async/await` интерфейс
- **Автоматические повторные попытки** — экспоненциальная задержка при ошибках
- **Ленивое подключение** — соединение создается при первом запросе

## Установка

```bash
pip install git+https://github.com/sidorov-works/vector_db_utils.git@v0.1.10
```

## Быстрый старт

```python
import asyncio
from vector_db_utils import VDBClient, VectorConfig, Chunk

async def main():
    # Создание клиента
    client = VDBClient(
        host="localhost",
        grpc_port=6334,
        grpc_enabled=True,
        scroll_point_limit=100
    )
    
    # Конфигурация векторов
    vectors_config = {
        "bge-small": VectorConfig(size=384, distance="Cosine"),
        "bge-large": VectorConfig(size=1024, distance="Cosine")
    }
    
    # Создание коллекции
    await client.create_collection("my_collection", vectors_config)
    
    # Вставка точек
    points = [{
        "id": "point_1",
        "vector": {
            "bge-small": [0.1, 0.2, ...],
            "bge-large": [0.3, 0.4, ...]
        },
        "payload": {"text": "Hello world", "tags": ["greeting"]}
    }]
    
    inserted = await client.upsert_points("my_collection", points)
    print(f"Inserted {inserted} points")
    
    # Поиск
    results = await client.search_by_tags_or(
        collection_name="my_collection",
        query_vector=[0.1, 0.2, ...],
        vector_name="bge-small",
        tenant_id="tenant_1",
        tags=["greeting"],
        limit=10
    )
    
    for result in results:
        print(f"Score: {result['score']}, Content: {result['payload']['text']}")
    
    # Закрытие
    await client.close()

asyncio.run(main())
```

## API

### VDBClient

#### Конструктор
```python
VDBClient(
    host: str = "localhost",
    grpc_port: int = 6334,
    grpc_enabled: bool = True,
    http_port: int = 6333,
    connection_timeout: float = 30.0,
    upsert_batch_size: int = 64,
    scroll_point_limit: int = 64
)
```

#### Операции с коллекциями
- `collection_exists(collection_name: str) -> bool`
- `create_collection(collection_name: str, vectors_config: Dict[str, VectorConfig]) -> bool`
- `delete_collection(collection_name: str) -> bool`
- `get_collection_info(collection_name: str) -> Optional[Dict]`
- `add_vectors_to_collection(collection_name: str, vectors_config: Dict[str, VectorParams]) -> bool`

#### Операции с точками
- `upsert_points(collection_name: str, points: List[Dict]) -> int`
- `upsert_chunks(collection_name: str, chunks: List[Chunk], document_name: str, tenant_id: str) -> int`
- `point_exists(collection_name: str, point_id: str) -> bool`
- `scroll_points(collection_name: str, filter: Optional[Filter] = None, offset=None) -> Tuple[List[Dict], Optional[Any]]`
- `delete_points_by_filter(collection_name: str, filter: Filter) -> bool`
- `remove_vector_completely(collection_name: str, vector_name: str) -> bool`

#### Поиск
- `search_by_tags_or(collection_name: str, query_vector: List[float], vector_name: str, tenant_id: str, tags: Optional[List[str]] = None, limit: int = 10, score_threshold: float = 0.5) -> List[Dict]`

#### Вспомогательные методы
- `build_tenant_filter(tenant_id: str) -> Filter`
- `build_document_filter(tenant_id: str, document_name: str) -> Filter`
- `build_tag_filter(tenant_id: str, tag: str) -> Filter`
- `build_tags_and_filter(tenant_id: str, tags: List[str]) -> Filter`
- `get_unique_field_values(collection_name: str, field_name: str, scroll_limit: Optional[int] = None) -> Tuple[set, int]`

#### Управление соединением
- `health_check() -> bool`
- `get_stats() -> Dict[str, Any]`
- `close() -> None`

### Модели

#### VectorConfig
```python
class VectorConfig(BaseModel):
    size: int
    distance: str  # "Cosine", "Euclid", "Dot"
```

#### Chunk
```python
class Chunk(BaseModel):
    content: str
    vectors: Dict[str, List[float]]
    title: Optional[str] = None
    topics: Optional[List[str]] = None
    original_index: Optional[int] = None
```

### Нормализация тегов
```python
from vector_db_utils import normalize_tag, normalize_tag_set

normalize_tag("My Tag!")        # "my_tag"
normalize_tag_set(["Tag1", "Tag2"])  # ["tag1", "tag2"]
```

## Требования

- Python >= 3.9
- `qdrant-client >= 1.17.0`
- `pydantic >= 2.12.5`

## Лицензия

MIT