import os
from modules.data_processor import process_pdfs

# Load environment-specific config
environment = os.getenv("ENV", "development")  # Default to 'development' if ENV is not set

if environment == "development":
    from config.dev_config import DevConfig as Config
else:
    from config.base_config import BaseConfig as Config

Config.CLAIM_LOCATION = "/Users/deveshsurve/UNIVERSITY/PROJECT/classify-pdf/data_files"

def main():
    path_to_process = Config.CLAIM_LOCATION + "/Compliance Report 1.pdf"
    print(f"Running {Config.APP_NAME} with {environment} configuration")
    process_pdfs(path_to_process)

if __name__ == "__main__":
    main()
