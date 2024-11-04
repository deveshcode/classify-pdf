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

# Add graph attributes for better styling
graph_attr = {
    "fontsize": "45",
    "splines": "ortho",    # Makes connections more organized with right angles
    "pad": "2.0"          # Adds more padding between clusters
}

# Add cluster attributes for consistent styling
cluster_attr = {
    "bgcolor": "#EBF3E7",
    "penwidth": "2",
    "fontsize": "12",
    "style": "rounded"    # Gives clusters rounded corners
}

with Diagram(
    "Healthcare Document Classification System V6",
    show=False,
    direction="LR",      # Left to right layout instead of top to bottom
    graph_attr=graph_attr
):
    # External Sources
    with Cluster("External Sources", graph_attr=cluster_attr):
        ehr = S3("EHR\nSystems")        # Using \n to make labels more compact
        scanner = S3("Document\nScanners")
        fax = S3("Fax\nSystems")
        manual = S3("Manual\nUpload")

    # Document Ingestion Service
    with Cluster("Document Ingestion Service", graph_attr=cluster_attr):
        validation = Lambda("Validation\nLayer")
        sanitization = Lambda("Sanitization\nLayer")
        encryption = Lambda("Encryption\nLayer")
        kms = KMS("Encryption\nKeys")
        
    # OCR Processing
    with Cluster("Document Processing", graph_attr=cluster_attr):
        ocr_service = Lambda("OCR Service")
        text_processing = Lambda("Text Processing")
        comprehend = Comprehend("NLP Processing")

    # Queue Management (Modified to have stage-specific queues)
    with Cluster("Queue Management", graph_attr=cluster_attr):
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
    with Cluster("LLM Processing", graph_attr=cluster_attr):
        llm_api = APIGateway("LLM API Gateway")
        llm_classifier = Lambda("LLM Classifier")
        iam = IAM("Access Control")
        llm_cache = Dynamodb("LLM Cache")

    # Storage Layer
    with Cluster("Storage", graph_attr=cluster_attr):
        doc_storage = S3("Document Storage")
        metadata_store = Dynamodb("Metadata Storage")
        results_store = Aurora("Results Storage")

    # Monitoring and Compliance
    with Cluster("Monitoring & Compliance", graph_attr=cluster_attr):
        cloudwatch = Cloudwatch("Monitoring")
        sns = SNS("Notifications")
        reporting = Lambda("Report Generation")
        audit_log = S3("Audit Logs")

    # Improve connection styling
    Edge.default_config = {
        "color": "#1a73e8",
        "style": "bold",
        "penwidth": "2"
    }

    # Define error paths with different color
    error_edge_attr = {
        "color": "#db4437",
        "style": "dashed",
        "penwidth": "1.5"
    }

    # Define connections with improved styling
    # Input flow
    [ehr, scanner, fax, manual] >> Edge(color="#4285f4") >> ingestion_event_bridge >> ingestion_queue
    ingestion_queue >> validation >> sanitization >> encryption
    ingestion_queue << Edge(**error_edge_attr) << ingestion_dlq
    
    # Document processing flow
    encryption >> processing_event_bridge >> processing_queue
    processing_queue >> ocr_service >> text_processing >> comprehend
    processing_queue << processing_dlq
    
    # Classification flow
    comprehend >> classification_event_bridge >> classification_queue
    classification_queue >> llm_api >> iam >> llm_classifier
    classification_queue << classification_dlq
    llm_classifier << llm_cache
    
    # Storage flow
    llm_classifier >> [doc_storage, metadata_store, results_store]
    
    # Monitoring flow
    [doc_storage, metadata_store, results_store] >> cloudwatch
    cloudwatch >> [sns, reporting, audit_log]