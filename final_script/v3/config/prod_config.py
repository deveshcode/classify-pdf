# config/prod_config.py

from .base_config import BaseConfig

class ProdConfig(BaseConfig):
    DEBUG = False
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@prod-db:5432/myapp")
    LOG_LEVEL = "WARNING"
