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

with Diagram("Healthcare Document Classification System V3", show=False):
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

    # Queue Management
    with Cluster("Queue Management"):
        event_bridge = Eventbridge("Event Router")
        batch_queue = SQS("Batch Queue")
        priority_queue = SQS("Priority Queue")
        standard_queue = SQS("Standard Queue")

    # LLM Processing
    with Cluster("LLM Processing"):
        llm_api = APIGateway("LLM API Gateway")
        llm_classifier = Lambda("LLM Classifier")
        iam = IAM("Access Control")

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

    # Define connections
    # Input flow
    [ehr, scanner, fax, manual] >> validation >> sanitization
    sanitization >> encryption
    encryption << kms
    
    # Document processing flow
    encryption >> ocr_service >> text_processing
    text_processing >> comprehend
    
    # Queue management flow
    comprehend >> event_bridge
    event_bridge >> [batch_queue, priority_queue, standard_queue]
    
    # LLM processing flow
    [batch_queue, priority_queue, standard_queue] >> llm_api
    llm_api >> iam >> llm_classifier
    
    # Storage flow
    llm_classifier >> [doc_storage, metadata_store, results_store]
    
    # Monitoring flow
    [doc_storage, metadata_store, results_store] >> cloudwatch
    cloudwatch >> sns
    cloudwatch >> reporting