from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser


class ThreadsConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.group_name = "threads_stream"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        user = self.scope.get("user")
        await self.channel_layer.group_send(
            self.group_name,
            {
                "type": "thread_event",
                "payload": {
                    "event": "presence",
                    "status": "online",
                    "user": getattr(user, "username", None) if getattr(user, "is_authenticated", False) else None,
                },
            },
        )

    async def disconnect(self, close_code):
        user = self.scope.get("user")
        await self.channel_layer.group_send(
            self.group_name,
            {
                "type": "thread_event",
                "payload": {
                    "event": "presence",
                    "status": "offline",
                    "user": getattr(user, "username", None) if getattr(user, "is_authenticated", False) else None,
                },
            },
        )
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive_json(self, content, **kwargs):
        # Echo ping/presence
        if content.get("type") == "ping":
            is_auth = not isinstance(self.scope.get("user"), AnonymousUser) and self.scope.get("user").is_authenticated
            await self.send_json({"type": "pong", "authenticated": bool(is_auth)})

    async def thread_event(self, event):
        # Broadcast thread/like/comment updates
        await self.send_json(event.get("payload", {}))


