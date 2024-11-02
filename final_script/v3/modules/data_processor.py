import os
from loguru import logger
from modules.database import save_processing_data
from modules.llm_classifier import LLMClassifier
from modules.data_cleaning import refined_clean_text
from modules.log_config import track_time
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from config.base_config import BaseConfig

DATABASE_URL = BaseConfig.DATABASE_URL
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def get_pdf_files(path):
    logger.info(f"Checking if path is a file or directory: {path}")
    if os.path.isfile(path) and path.endswith(".pdf"):
        logger.info(f"Path is a PDF file: {path}")
        return [path]
    elif os.path.isdir(path):
        logger.info(f"Path is a directory: {path}")
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pdf")]
    else:
        logger.error(f"Provided path is neither a PDF file nor a directory containing PDFs: {path}")
        raise ValueError("Provided path is neither a PDF file nor a directory containing PDFs.")

@track_time
def extract_text_ocr(pdf_file):
    logger.info(f"Extracting text from PDF file: {pdf_file}")
    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path(pdf_file, dpi=300)
    logger.info(f"Converted PDF to {len(images)} images")
    text = ''.join(pytesseract.image_to_string(image) + "\n" for image in images)
    logger.info(f"Extracted text from image successfully")
    return text

@track_time
def process_pdfs(path):
    pdf_files = get_pdf_files(path)
    classifier = LLMClassifier()

    for pdf_file in pdf_files:
        logger.info(f"Processing file: {pdf_file}")
        file_name = os.path.basename(pdf_file)
        file_location = pdf_file
        process_metadata = {}

        raw_text, ocr_time = extract_text_ocr(pdf_file)
        process_metadata["OCR"] = {"time": ocr_time}

        cleaned_text, clean_time = refined_clean_text(raw_text)
        process_metadata["Text Cleaning"] = {"time": clean_time}

        (predicted_class, confidence, high_conf_classes, classify_cost), classify_time = classifier.classify_document(cleaned_text, file_name)
        process_metadata["Classification"] = {"time": classify_time, "cost": classify_cost}

        save_processing_data(
            file_name,
            file_location,
            raw_text,
            cleaned_text,
            predicted_class,
            confidence,
            process_metadata,
            high_conf_classes
        )
        logger.info(f"File: {file_location}, Predicted Class: {predicted_class}, Confidence: {confidence}")
        logger.info(f"Process Metadata: {process_metadata}")
