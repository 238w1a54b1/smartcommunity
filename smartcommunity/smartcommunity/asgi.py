"""
ASGI config for smartcommunity project with Channels routing.
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartcommunity.smartcommunity.settings')

django_asgi_app = get_asgi_application()

from app import consumers  # noqa: E402

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path("ws/threads/", consumers.ThreadsConsumer.as_asgi()),
        ])
    ),
})
