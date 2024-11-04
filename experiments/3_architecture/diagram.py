from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import Lambda
from diagrams.aws.integration import SQS, SNS
from diagrams.aws.network import APIGateway
from diagrams.aws.storage import S3
from diagrams.aws.ml import Textract
from diagrams.aws.management import Cloudwatch

# Define your diagram
with Diagram("Healthcare Architecture - AWS", show=False):
    # External sources cluster
    with Cluster("External Sources"):
        ehr = S3("EHR Systems")
        doc_scanner = S3("Document Scanners")
        fax = S3("Fax Systems")
        manual_upload = S3("Manual Upload")

    # Document ingestion services
    with Cluster("Document Ingestion Service"):
        validation = Lambda("Validation Layer")
        sanitization = Lambda("Sanitization Layer")
        encryption = Lambda("Encryption Layer")

    # OCR Processing
    ocr_service = Textract("OCR Service")

    # Processing Layer
    with Cluster("Processing Layer"):
        text_processing = Lambda("Text Processing")
        llm_classifier = Lambda("LLM Classifier")

    # Storage Layer
    with Cluster("Storage"):
        doc_storage = S3("Document Storage")
        metadata_storage = S3("Metadata Storage")
        results_storage = S3("Results Storage")

    # Queue Management
    with Cluster("Queue Management"):
        batch_queue = SQS("Batch Queue")
        priority_queue = SQS("Priority Queue")
        standard_queue = SQS("Standard Queue")

    # Output Layer
    with Cluster("Output Layer"):
        api_gateway = APIGateway("API Gateway")
        notification = SNS("Notification Service")
        reporting = Lambda("Reporting Service")

    # Monitoring and Compliance
    monitoring = Cloudwatch("Monitoring")

    # Define data flow - improved connections
    # Connect all input sources
    ehr >> validation
    doc_scanner >> validation
    fax >> ocr_service
    manual_upload >> validation

    # Main processing flow
    validation >> sanitization >> encryption
    encryption >> batch_queue
    encryption >> priority_queue
    encryption >> standard_queue

    # OCR and processing flow
    ocr_service >> text_processing
    
    # Queue processing
    batch_queue >> text_processing
    priority_queue >> text_processing
    standard_queue >> text_processing

    # Classification and storage
    text_processing >> llm_classifier
    llm_classifier >> [metadata_storage, results_storage]
    
    # Output flows
    llm_classifier >> api_gateway
    api_gateway >> notification
    notification >> monitoring
    
    # Reporting flow
    reporting << metadata_storage
    reporting << results_storage
