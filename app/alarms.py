from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from app.database import alarms_collection
from app.schemas import AlarmCreate, AlarmRead


def _read(document: dict) -> AlarmRead:
    return AlarmRead(id=str(document["_id"]), source=document["source"], severity=document["severity"], message=document["message"], status=document["status"], created_at=document["created_at"], acknowledged_at=document.get("acknowledged_at"))


async def create_alarm(payload: AlarmCreate) -> AlarmRead:
    document = {**payload.model_dump(mode="json"), "status": "active", "created_at": datetime.now(timezone.utc), "acknowledged_at": None}
    document["_id"] = (await alarms_collection.insert_one(document)).inserted_id
    return _read(document)


async def list_alarms(limit: int = 100) -> list[AlarmRead]:
    cursor = alarms_collection.find().sort("created_at", -1).limit(limit)
    return [_read(document) async for document in cursor]


async def get_alarm(alarm_id: str) -> AlarmRead | None:
    try:
        document = await alarms_collection.find_one({"_id": ObjectId(alarm_id)})
    except InvalidId:
        return None
    return _read(document) if document else None


async def acknowledge_alarm(alarm_id: str) -> AlarmRead | None:
    try:
        object_id = ObjectId(alarm_id)
    except InvalidId:
        return None
    await alarms_collection.update_one({"_id": object_id, "status": "active"}, {"$set": {"status": "acknowledged", "acknowledged_at": datetime.now(timezone.utc)}})
    document = await alarms_collection.find_one({"_id": object_id})
    return _read(document) if document else None
