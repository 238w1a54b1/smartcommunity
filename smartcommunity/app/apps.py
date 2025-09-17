# app/apps.py
from django.apps import AppConfig

class UsersConfig(AppConfig):   # <-- Change AppConfig to UsersConfig
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'smartcommunity.app'

    def ready(self):
        from . import signals
  # Make sure this points to your signals
