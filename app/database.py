from typing import Any
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from app.config import get_settings

settings = get_settings()
client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
    settings.mongodb_url.get_secret_value(), appname="patient-watch", serverSelectionTimeoutMS=5000
)
alarms_collection: AsyncCollection[dict[str, Any]] = client[settings.mongodb_database]["alarms"]
images_collection: AsyncCollection[dict[str, Any]] = client[settings.mongodb_database]["images"]
analyses_collection: AsyncCollection[dict[str, Any]] = client[settings.mongodb_database]["vlm_analyses"]


async def init_database() -> None:
    await client.admin.command("ping")
    await alarms_collection.create_index([("created_at", DESCENDING)])
    await alarms_collection.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
    await images_collection.create_index([("created_at", DESCENDING)])
    await images_collection.create_index([("processed", ASCENDING), ("created_at", DESCENDING)])
    await analyses_collection.create_index([("image_id", ASCENDING)], unique=True)
    await analyses_collection.create_index([("created_at", DESCENDING)])


async def close_database() -> None:
    await client.close()
