from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import Glue, Athena
from diagrams.aws.compute import Lambda, EC2
from diagrams.aws.database import Dynamodb, RDS, Aurora
from diagrams.aws.integration import SQS, SNS
from diagrams.aws.management import Cloudwatch
from diagrams.aws.ml import Comprehend, Rekognition, Translate
from diagrams.aws.security import IAM
from diagrams.aws.storage import S3

with Diagram("Healthcare Document Classification System", show=False):

    with Cluster("Data Ingestion Layer"):
        external_sources = [
            Glue("Document Extraction Service"),
            Athena("Data Lake"),
        ]
        s3_data_lake = S3("Document Storage")

    with Cluster("Processing Layer"):
        lambda_ocr = Lambda("OCR Service")
        comprehend_nlp = Comprehend("NLP Processing")
        rekognition = Rekognition("Image Analysis")
        translate = Translate("Translation")

    with Cluster("Queue Management"):
        batch_queue = SQS("Batch Queue")
        priority_queue = SQS("Priority Queue")
        standard_queue = SQS("Standard Queue")

    with Cluster("Processing"):
        data_classifier = Lambda("LLM Classifier")
        data_analyzer = EC2("Data Analyzer")

    with Cluster("Storage"):
        metadata_store = Dynamodb("Metadata Storage")
        results_store = RDS("Results Database")
        data_store = Aurora("Document DB")

    with Cluster("Monitoring and Compliance"):
        cloudwatch_logs = Cloudwatch("Logging")
        cloudwatch_metrics = Cloudwatch("Metrics")
        sns_notifications = SNS("Notifications")

    with Cluster("Output"):
        api_gateway = EC2("API Gateway")
        reporting_service = Lambda("Reporting Service")

    # Define Connections
    for source in external_sources:
        source >> Edge(label="Extract Data") >> lambda_ocr
    lambda_ocr >> Edge(label="OCR") >> comprehend_nlp >> data_classifier >> data_analyzer
    data_classifier >> batch_queue
    data_classifier >> priority_queue
    data_classifier >> standard_queue

    batch_queue >> data_analyzer >> data_store
    priority_queue >> data_analyzer >> metadata_store
    standard_queue >> data_analyzer >> results_store

    data_store >> cloudwatch_logs
    metadata_store >> cloudwatch_metrics
    results_store >> sns_notifications

    api_gateway << sns_notifications
    reporting_service
