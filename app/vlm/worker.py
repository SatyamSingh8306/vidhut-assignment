"""
ImageWorker — stores camera frames in MongoDB and processes them with the VLM.

Capture loop: saves a frame every `capture_interval` seconds (and on demand).
Processing loop: picks unprocessed images, runs VLM analysis, stores the result
and publishes `image` / `analysis` events to connected consoles.
Media lives in MongoDB only; raw video/audio is never recorded elsewhere.
"""

import asyncio
import logging
from datetime import datetime, timezone

from bson import ObjectId

from app.database import images_collection, analyses_collection
from app.events import event_hub
from app.vlm.analyzer import VLMAnalyzer, analysis_document

logger = logging.getLogger(__name__)


class ImageWorker:
    def __init__(self, analyzer: VLMAnalyzer, frame_provider, capture_interval: float = 10.0) -> None:
        """
        frame_provider: callable returning latest JPEG bytes or None (VideoMonitor.latest_frame).
        capture_interval: seconds between automatic captures.
        """
        self.analyzer = analyzer
        self.frame_provider = frame_provider
        self.capture_interval = capture_interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._wakeup: asyncio.Event = asyncio.Event()

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run(), name="image-worker")

    async def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def request_capture(self) -> None:
        """Wake the worker for an immediate capture + analysis."""
        self._wakeup.set()

    async def _run(self) -> None:
        logger.info("Image worker started (interval=%.1fs, vlm=%s)", self.capture_interval, self.analyzer.ready)
        while self._running:
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self.capture_interval)
            except asyncio.TimeoutError:
                pass
            self._wakeup.clear()
            try:
                await self.capture_and_process()
            except Exception:
                logger.exception("Image worker cycle failed")
            await asyncio.sleep(1.0)

    async def capture_and_process(self) -> dict | None:
        jpeg = self.frame_provider()
        if not jpeg:
            return None
        image = {
            "data": jpeg,
            "created_at": datetime.now(timezone.utc),
            "source": "camera",
            "help_detected": False,
            "processed": False,
        }
        result = await images_collection.insert_one(image)
        image["_id"] = result.inserted_id
        payload = {
            "id": str(result.inserted_id),
            "created_at": image["created_at"].isoformat(),
            "source": image["source"],
            "processed": False,
        }
        await event_hub.publish("image", payload)
        await self.process_image(str(result.inserted_id), jpeg, image)
        return {"id": str(result.inserted_id)}

    async def process_image(self, image_id: str, jpeg: bytes, image_doc: dict | None = None) -> dict | None:
        """Run VLM analysis on an image and persist the result."""
        try:
            existing = await analyses_collection.find_one({"image_id": image_id})
            if existing:
                return existing
            if not self.analyzer.ready:
                return None
            analysis = await self.analyzer.analyze(jpeg)
            doc = analysis_document(image_id, analysis)
            await analyses_collection.insert_one(doc)
            await images_collection.update_one(
                {"_id": ObjectId(image_id)},
                {"$set": {"processed": True, "help_detected": analysis.get("requires_attention", False)}},
            )
            payload = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in doc.items()}
            payload["id"] = payload.pop("image_id")
            await event_hub.publish("analysis", payload)
            if doc.get("requires_attention"):
                logger.info("VLM flagged image %s (intent=%s)", image_id, doc["intent"])
            return doc
        except Exception:
            logger.exception("VLM analysis failed for image %s", image_id)
            return None

    async def process_pending(self, limit: int = 5) -> int:
        """Process up to `limit` unprocessed images (used at startup/backfill)."""
        cursor = images_collection.find({"processed": False}).sort("created_at", -1).limit(limit)
        count = 0
        async for doc in cursor:
            if await self.process_image(str(doc["_id"]), doc["data"], doc):
                count += 1
        return count
