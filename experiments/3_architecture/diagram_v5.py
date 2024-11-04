from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import Glue, Athena
from diagrams.aws.compute import Lambda, EC2
from diagrams.aws.database import Dynamodb, RDS, Aurora
from diagrams.aws.integration import SQS, SNS
from diagrams.aws.integration import Eventbridge
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Comprehend, Rekognition, Translate
from diagrams.aws.network import APIGateway
from diagrams.aws.security import IAM, KMS
from diagrams.aws.storage import S3

with Diagram("Healthcare Document Classification System V7", show=False):
    # External Sources
    with Cluster("External Sources"):
        ehr = S3("EHR Systems")
        scanner = S3("Document Scanners")
        fax = S3("Fax Systems")
        manual = S3("Manual Upload")

    # Document Ingestion Service
    with Cluster("Document Ingestion Service"):
        validation = Lambda("Validation Layer")
        sanitization = Lambda("Sanitization Layer")
        encryption = Lambda("Encryption Layer")
        kms = KMS("Encryption Keys")
        
    # OCR Processing
    with Cluster("Document Processing"):
        ocr_service = Lambda("OCR Service")
        text_processing = Lambda("Text Processing")
        comprehend = Comprehend("NLP Processing")

    # Queue Management (Modified to have stage-specific queues)
    with Cluster("Queue Management"):
        # Ingestion Queues
        ingestion_event_bridge = Eventbridge("Ingestion Router")
        ingestion_queue = SQS("Ingestion Queue")
        ingestion_dlq = SQS("Ingestion DLQ")
        
        # Processing Queues
        processing_event_bridge = Eventbridge("Processing Router")
        processing_queue = SQS("Processing Queue")
        processing_dlq = SQS("Processing DLQ")
        
        # Classification Queues
        classification_event_bridge = Eventbridge("Classification Router")
        classification_queue = SQS("Classification Queue")
        classification_dlq = SQS("Classification DLQ")

    # LLM Processing
    with Cluster("LLM Processing"):
        llm_api = APIGateway("LLM API Gateway")
        llm_classifier = Lambda("LLM Classifier")
        iam = IAM("Access Control")
        llm_cache = Dynamodb("LLM Cache")

    # Storage Layer
    with Cluster("Storage"):
        doc_storage = S3("Document Storage")
        metadata_store = Dynamodb("Metadata Storage")
        results_store = Aurora("Results Storage")

    # Monitoring and Compliance
    with Cluster("Monitoring & Compliance"):
        cloudwatch = Cloudwatch("Monitoring")
        sns = SNS("Notifications")
        reporting = Lambda("Report Generation")
        audit_log = S3("Audit Logs")

    # Define connections with improved styling
    # Input flow
    [ehr, scanner, fax, manual] >> Edge(color="#4285f4", label="1") >> ingestion_event_bridge >> Edge(label="2") >> ingestion_queue
    ingestion_queue >> Edge(label="3") >> validation >> Edge(label="4") >> sanitization >> Edge(label="5") >> encryption
    ingestion_queue <<  ingestion_dlq
    
    # Document processing flow
    encryption >> Edge(label="6") >> processing_event_bridge >> Edge(label="7") >> processing_queue
    processing_queue >> Edge(label="8") >> ocr_service >> Edge(label="9") >> text_processing >> Edge(label="10") >> comprehend
    processing_queue << processing_dlq
    
    # Classification flow
    comprehend >> Edge(label="11") >> classification_event_bridge >> Edge(label="12") >> classification_queue
    classification_queue >> Edge(label="13") >> llm_api >> Edge(label="14") >> iam >> Edge(label="15") >> llm_classifier
    classification_queue <<  classification_dlq
    llm_classifier << Edge(label="16") << llm_cache
    
    # Storage flow
    llm_classifier >> Edge(label="17") >> [doc_storage, metadata_store, results_store]
    
    # Monitoring flow
    [doc_storage, metadata_store, results_store] >> Edge(label="18") >> cloudwatch
    cloudwatch >> Edge(label="19") >> [sns, reporting, audit_log]