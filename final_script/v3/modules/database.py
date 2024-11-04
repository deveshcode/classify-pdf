import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from loguru import logger
from ..config.base_config import BaseConfig

DATABASE_URL = BaseConfig.DATABASE_URL
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    file_name = Column(String, unique=True)
    file_location = Column(Text)
    raw_text = Column(Text)
    cleaned_text = Column(Text)
    classified_category = Column(String)
    confidence = Column(Float)
    high_confidence_classes = Column(Text)
    process_metadata = Column(Text)
    ground_truth = Column(String)  # Store the ground truth label

Base.metadata.create_all(engine)

def save_processing_data(file_name, file_location, raw_text, cleaned_text, classified_category, confidence, metadata, high_conf_classes):
    process_metadata_json = json.dumps(metadata)
    high_conf_classes_json = json.dumps(high_conf_classes)

    doc = session.query(Document).filter_by(file_name=file_name).first()
    if not doc:
        doc = Document(
            file_name=file_name,
            file_location=file_location,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            classified_category=classified_category,
            confidence=confidence,
            process_metadata=process_metadata_json,
            high_confidence_classes=high_conf_classes_json
        )
        session.add(doc)
    else:
        doc.raw_text = raw_text
        doc.cleaned_text = cleaned_text
        doc.classified_category = classified_category
        doc.confidence = confidence
        doc.process_metadata = process_metadata_json
        doc.high_confidence_classes = high_conf_classes_json
    session.commit()
