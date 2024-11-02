# config/dev_config.py

from .base_config import BaseConfig
import os

class DevConfig(BaseConfig):
    DEBUG = True
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///dev.db")
    LOG_LEVEL = "DEBUG"
