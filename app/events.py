from fastapi import WebSocket
from typing import Any


class EventHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, socket: WebSocket) -> None:
        await socket.accept()
        self._clients.add(socket)

    def disconnect(self, socket: WebSocket) -> None:
        self._clients.discard(socket)

    async def publish(self, event: str, data: dict[str, Any]) -> None:
        payload = {"event": event, "data": data}
        dead = []
        for socket in self._clients:
            try:
                await socket.send_json(payload)
            except RuntimeError:
                dead.append(socket)
        for socket in dead:
            self.disconnect(socket)


event_hub = EventHub()
