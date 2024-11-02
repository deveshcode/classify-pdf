# config/base_config.py

import os
import dotenv

dotenv.load_dotenv()

class BaseConfig:
    APP_NAME = "MyApp"
    DEBUG = False
    LLM_COST_PER_TOKEN = 0.0004
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
    API_TIMEOUT = 30  # seconds
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///results_v1.db")