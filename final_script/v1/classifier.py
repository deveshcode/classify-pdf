# Let's load all the pdf files from the directory

import os
import dotenv

dotenv.load_dotenv()

folder_path = os.getenv("CLAIM_LOCATION")
pdf_files = [folder_path + "/" + f for f in os.listdir(folder_path) if f.endswith(".pdf")]

# def extract_text_ocr(pdf_file):
#     from pdf2image import convert_from_path
#     import pytesseract
#     images = convert_from_path(pdf_file, dpi=300)
#     text = pytesseract.image_to_string(images[0])
#     return text

# text = extract_text_ocr(pdf_files[1])   

import re

def refined_clean_text(text):
    # Step 1: Remove unwanted special characters and symbols at the start
    text = re.sub(r'[^\w\s:/.,%-]+', '', text)
    
    # Step 2: Replace specific lingering artifacts (such as OCR errors)
    text = re.sub(r'\boor\b|\be\b|\baye\b|\beee\b', '', text)  # Remove specific unwanted words
    
    # Step 3: Remove repeated punctuation and excessive whitespace
    text = re.sub(r'\n+', '\n', text)  # Convert multiple newlines to a single newline
    text = re.sub(r'\s{2,}', ' ', text)  # Convert multiple spaces to a single space
    
    # Step 4: Add space after punctuation where missing
    text = re.sub(r'(?<=[.,])(?=\S)', ' ', text)  # Add space after commas and periods where needed
    
    # Step 5: Remove text following unwanted headers or patterns (for contact info, unwanted lines)
    contact_info_pattern = r'(Phone|Fax|Email):? [^\n]*\n'
    text = re.sub(contact_info_pattern, '', text)
    
    # Step 6: Remove any lingering symbols or non-word characters at the start of the text
    text = re.sub(r'^[^\w]+', '', text)

    # Step 7: Add space between numbers and units or percentages if missing
    text = re.sub(r'(\d)(cmH20|L/min|%)', r'\1 \2', text)  # Add space between number and units/percentage

    # Step 8: Ensure single space around punctuation (remove spaces before, add after if needed)
    text = re.sub(r'\s+([:,])', r'\1', text)  # Remove spaces before colons and commas
    text = re.sub(r'([:,])\s+', r'\1 ', text)  # Ensure single space after colons and commas

    # Step 9: Final trim of leading and trailing spaces
    text = text.strip()
    
    return text

# Apply cleaning function to each entry in labeled_results
cleaned_text = refined_clean_text(text)

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import os
from typing import Dict, Tuple, List
import pandas as pd
from sklearn.metrics import classification_report
from litellm import completion
import dotenv

dotenv.load_dotenv()    

class LLMClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini", threshold: float = 0.5):
        """
        Initialize the document classifier with LLM support.

        Args:
            model_name: Name of the model to use
            threshold: Minimum confidence threshold for classification
        """
        self.model_name = model_name
        self.threshold = threshold
        self.label_prompts = self.create_class_prompts()
        self.examples = self.create_few_shot_examples()  # Full examples to use selectively

    def create_class_prompts(self) -> Dict[str, str]:
        """
        Define a prompt for each class to guide the LLM in identifying document type.
        """
        return {
            "Compliance": "This document should include patient compliance information regarding medical device usage and therapy adherence. It should have terms like 'compliance percentage,' 'usage hours,' 'therapy effectiveness metrics,' or similar indicators of device usage compliance.",
            "Sleep": "This document should contain results from a clinical sleep study, such as polysomnography. Expected content includes terms like 'sleep patterns,' 'respiratory events,' 'sleep efficiency,' 'polysomnography,' and other sleep metrics.",
            "Order": "This document should be a medical equipment or supply order form, containing details about items ordered, patient and provider information, or purchase requisition details.",
            "Delivery": "This document is a confirmation of delivery for medical equipment or supplies. It may include terms like 'delivery receipt,' 'equipment delivered,' 'delivery confirmation,' or 'tracking details.'",
            "Physician": "This document contains physician notes from a patient consultation or examination, likely including medical assessments, observations, treatment plans, and clinical findings.",
            "Prescription": "This document is a medical prescription, detailing medication names, dosages, refills, and instructions for medication usage."
        }

    def create_few_shot_examples(self) -> Dict[str, str]:
        """
        Create full few-shot examples for each class to use selectively.
        """
        return {
            "Compliance": """
            Document: "ResMed AirView Compliance Report PATIENT INFO: Name: John Doe, ID: 987654321, DOB: 04/25/1968. COMPLIANCE PERIOD: 09/01/2023 - 09/30/2023. COMPLIANCE SUMMARY: Compliance Met: Yes, Compliance Percentage: 85%. USAGE DETAILS: Total Usage Days: 25/30 days (83%), Days with ≥ 4 hours usage: 23 days (77%). Total Usage Hours: 176 hours 45 minutes. Average Usage (All Days): 5 hours 54 minutes, Average Usage (Days Used): 7 hours 21 minutes. Median Usage (Days Used): 7 hours 30 minutes. DEVICE INFO: Device Model: AirSense 11 AutoSet, Serial Number: 22334455667. SETTINGS: Mode: AutoSet, Min Pressure: 6 cmH2O, Max Pressure: 15 cmH2O, EPR: Fulltime, EPR Level: 2. PERFORMANCE METRICS: Leak Rate - Median: 12.4 L/min, 95th Percentile: 28.7 L/min, Max: 35.1 L/min. Events Per Hour - AI: 2.2, HI: 1.3, AHI: 3.5. Apnea Index - Central: 0.8, Obstructive: 1.4, Unknown: 0.0. RECOMMENDATION: Continue current settings; follow up in 3 months to re-evaluate compliance and usage patterns."
            Class: Compliance_Report
            Confidence: High
            """,
            
            "Sleep": """
            Document: "Date: 05/12/2023 FAX No: 123-456-7890 Institute of Sleep Medicine, Sleep Study Report. PATIENT INFO: Name: Jane Doe, ID: 987654, Age: 52, Gender: Female, BMI: 28.4. TEST DATE: 05/10/2023. REASON FOR STUDY: Evaluate for obstructive sleep apnea due to symptoms of daytime sleepiness and loud snoring. STUDY DETAILS: Overnight polysomnography was conducted, including continuous monitoring of airflow, respiratory effort, heart rate, and oxygen saturation. RESULTS SUMMARY: Total recording time: 7 hours, with sleep time of 5 hours 15 minutes. Apnea-hypopnea index (AHI) calculated at 22.5, indicating moderate obstructive sleep apnea. Lowest oxygen saturation observed at 84%. DIAGNOSIS & RECOMMENDATIONS: Diagnosis confirmed as moderate obstructive sleep apnea. Recommended treatment includes CPAP therapy with an initial pressure setting of 8 cmH2O, to be reviewed after 4 weeks. Patient advised on sleep hygiene practices and to avoid sedatives and alcohol."
            Class: Sleep_Study_Report
            Confidence: High
            """,

            "Order": """
            Document: "DATE: 08/15/2023 FAX: 555-123-4567 Medical Equipment Order Form PATIENT INFORMATION: Name: Alice Johnson, DOB: 06/10/1965, ID: 123456789. ORDER DETAILS: ORDER DATE: 08/14/2023. DIAGNOSIS: Obstructive Sleep Apnea (ICD-10 G47.33). EQUIPMENT REQUIRED: - ResMed AirSense 10 CPAP Machine, - Heated Humidifier (Code: E0562), - Full Face Mask (Code: A7030), - SlimLine Tubing (Code: A7037, 1 per 3 months), - Disposable Filters (Code: A7038, 2 per month). SUPPLY FREQUENCY: Filters to be replaced monthly, tubing every 3 months, mask every 6 months. NOTES: Please schedule initial setup within 3 business days. Fax confirmation of order receipt and anticipated setup date to 555-987-6543. Signature: Dr. Henry Lewis, NPI: 1234567890, Date: 08/15/2023."
            Class: Order
            Confidence: High
            """,

            "Delivery": """
            Document: "DELIVERY RECEIPT Provider: Care Medical Supplies Inc. Delivery Location: 1234 Main St., Suite 500, Orlando, FL 32801, Phone: 407-555-6789. DELIVERY DETAILS: Delivery Date: 09/15/2023, CSR: Jane Doe. PATIENT INFO: Name: James Smith, DOB: 03/14/1956, Account Number: 456789123. Insurance: UnitedHealthcare, HIPAA Signature on file: Yes. DELIVERY ITEMS: - Item: CPAP Machine, Model: AirSense 10, Serial Number: 987654321, Quantity: 1, Type: Rental, Unit Price: $65.00, Total: $65.00. - Item: Heated Humidifier, Code: E0562, Quantity: 1, Type: Purchase, Unit Price: $150.00, Total: $150.00. - Item: Nasal Mask, Model: ComfortGel Blue, Size: Medium, Quantity: 1, Type: Purchase, Unit Price: $60.00, Total: $60.00. - Item: Tubing, Model: SlimLine, Quantity: 1, Type: Purchase, Unit Price: $20.00, Total: $20.00. PAYMENT DETAILS: Payment Method: Insurance Billing, Co-pay: $0.00, Total Amount Due: $295.00. SPECIAL INSTRUCTIONS: Ensure delivery confirmation signature is obtained. Initial setup instructions provided to patient."
            Class: Delivery_Ticket
            Confidence: High
            """,

            "Physician": """
            Document: "7/17/2023 10:20 AM FROM: Page 1 of 17\n1. Follow\nNye\nup: Sleep stucy results\nTaking\n6 Bisacodyl 5 MG Tablet Delayed\nRelease 1 tablet as needed Orally\nOnce a day Gemfibrozil 600 MG Tablet"
            Class: Physician_Notes
            Confidence: High
            """,

            "Prescription": """
            Document: "HCMG Pulmonary Date: Apr 27, 2023\n4725 N Federal Hwy, Ste 203\nFort Lauderdale FL 33308-4603\nCPAP DME Order ID: 701783674\nOrder Date: 4/27/2023\nDiagnosis: Obstructive sleep apnea, adult G47.33\nQuantity: 1\nHeight: Weight: Scheduling Instructions: If the AHI or RDI is calculated based on less than 2 hours of sleep or recording time, the total number of recorded events used to calculate the AHI or RD must be at least the number of events that would have been required in a 2 hour period.\nThe face-to-face evaluation was completed by: Reasons for Script: New prescription AND ALL RELATED SUPPLIES\nAdditional providers who completed a face to face evaluation of the patient: SORHAGE, FRANK 50595\nChanges Only: AUTO UNIT L5/H20\nTubing: A7037 Reusable tubing 1/3mo\nTubing: A4604 Heated tubing 1/3mo\nType of Interface and Accessories: A7038 Disposable filter 2/1mo MASK FIT TO PATIENTS COMFORT\nType of Interface and Accessories: A7039 Reusable filter 1/6mo\nType of interface and Accessories: A4604 Heated tubing 1/3mo\nType of interface and Accessories: A7046 Humidifier Chamber 1/6mo\nLength of Need 12 Months: 99\nThe AHI is from 5 to 14 events per hour with documented symptoms of: OSA G47.33\nINSURANCE PAYOR PLAN GROUP SUBSCRIBER ID\nPrimary: MEDICARE 40000101\nSecondary: AARP 10000701\nElectronically Signed by\non Apr 27, 2023, at 5:18"
            Class: Prescription
            Confidence: High
            """
        }


    def classify_document(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a document using LLM and determine if it belongs to exactly one class.

        Args:
            text: Document text to classify

        Returns:
            Tuple of (predicted_class, confidence, all_scores)
        """
        scores = {}
        for label, prompt in self.label_prompts.items():
            response = completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"Identify if the following document matches the description: {prompt}. Return only Yes/No and confidence in percentage format."},
                    {"role": "user", "content": text}
                ]
            )
            score = self.extract_confidence(response)
            scores[label] = score

        # Filter classes with high confidence ("Yes" responses)
        high_confidence_classes = {label: conf for label, conf in scores.items() if conf >= self.threshold}
        
        # Check for unique high-confidence classification
        if len(high_confidence_classes) == 1:
            predicted_class = next(iter(high_confidence_classes))
            if "Physician" in predicted_class:
                predicted_class = "Physician"
            elif "Prescription" in predicted_class:
                predicted_class = "Prescription"
            elif "Delivery" in predicted_class:
                predicted_class = "Delivery"
            elif "Sleep" in predicted_class:
                predicted_class = "Sleep"
            elif "Compliance" in predicted_class:
                predicted_class = "Compliance"
            elif "Order" in predicted_class:
                predicted_class = "Order"
            confidence = high_confidence_classes[predicted_class]
            print("Predicted Class: ", predicted_class, "Confidence: ", confidence)
            return predicted_class, confidence, scores
        elif len(high_confidence_classes) == 0:
            # Use few-shot example classification with all classes
            high_conf_classes = {label: 0.0 for label in self.label_prompts.keys()}
            return self.classify_with_few_shot(text, high_conf_classes)
        elif len(high_confidence_classes) > 1:
            # Use few-shot example classification
            return self.classify_with_few_shot(text, high_confidence_classes)
        else:
            return "notsure", 0.0, scores

    def classify_with_few_shot(self, text: str, high_conf_classes: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify document using few-shot examples for high-confidence classes.

        Args:
            text: Document text to classify
            high_conf_classes: Dictionary of high-confidence classes

        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Generate few-shot examples for high-confidence classes
        few_shot_examples = "\n\n".join(self.examples[label] for label in high_conf_classes.keys())
        
        prompt = f"""
        {few_shot_examples}
        
        Classify the following document into one of these classes: {', '.join(high_conf_classes.keys())}.
        Document: "{text}"
        """

        # Make the single API call for this document with few-shot examples
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse the response to extract the predicted class
        response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # predicted_class, confidence = self.extract_class_and_confidence(response_text)
        if "Physician" in response_text:
            predicted_class = "Physician"
        elif "Prescription" in response_text:
            predicted_class = "Prescription"
        elif "Delivery" in response_text:
            predicted_class = "Delivery"
        elif "Sleep" in response_text:
            predicted_class = "Sleep"
        elif "Compliance" in response_text:
            predicted_class = "Compliance"
        elif "Order" in response_text:
            predicted_class = "Order"

        return predicted_class, 0.9, high_conf_classes

    def extract_confidence(self, response: dict) -> float:
        """
        Extract confidence level from LLM response.

        Args:
            response: LLM response dictionary

        Returns:
            Extracted confidence score as a float (0-1 range).
        """
        response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
        if "yes" in response_text.lower():
            confidence = float((response_text.split("%")[0].strip()).split(" ")[-1]) / 100
        else:
            confidence = 0.0
        return confidence

    def extract_class_and_confidence(self, response_text: str) -> Tuple[str, float]:
        """
        Parse LLM response to extract the predicted class and confidence.

        Args:
            response_text: LLM response in text format

        Returns:
            Tuple of (predicted_class, confidence_score as a float)
        """
        predicted_class = "Unknown"
        confidence_score = 0.0
        lines = response_text.splitlines()
        for line in lines:
            if "Class:" in line:
                predicted_class = line.split("Class:")[1].strip()
            elif "Confidence:" in line:
                confidence_score = float(line.split("Confidence:")[1].replace("%", "").strip())
        
        return predicted_class, confidence_score

    def evaluate_classification(self, test_data: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        Evaluate classification performance on test data, including misclassification report,
        AUC scores, and ROC curve.

        Args:
            test_data: Dictionary of test documents with their true labels

        Returns:
            DataFrame with evaluation metrics and misclassified documents
        """
        predictions = []
        true_labels = []
        confidences = []
        misclassified_docs = []  # Track misclassified documents

        for doc_name, doc_data in test_data.items():
            pred_class, confidence, _ = self.classify_document(doc_data['text'])
            print(f"Document: {doc_name} - Classified as: {pred_class} with confidence: {confidence}%\n")
            predictions.append(pred_class)
            true_labels.append(doc_data['label'])
            confidences.append(confidence)

            # Log misclassifications
            if pred_class != doc_data['label']:
                misclassified_docs.append({
                    "Document Name": doc_name,
                    "True Class": doc_data['label'],
                    "Predicted Class": pred_class
                })

        # Generate classification report
        report = classification_report(true_labels, predictions, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        # Add average confidence per class
        confidence_series = pd.Series(confidences, index=true_labels)
        avg_confidences = confidence_series.groupby(level=0).mean()
        df_report['avg_confidence'] = avg_confidences

        # Display misclassified documents
        misclassified_df = pd.DataFrame(misclassified_docs)
        if not misclassified_df.empty:
            print("\nMisclassified Documents:\n", misclassified_df)

        # Calculate and display ROC AUC
        lb = LabelBinarizer()
        true_labels_binary = lb.fit_transform(true_labels)
        predictions_binary = lb.transform(predictions)
        roc_auc = roc_auc_score(true_labels_binary, predictions_binary, average="macro", multi_class="ovr")
        print("\nROC AUC Score:", roc_auc)
        
        # Plot ROC curve for each class
        self.plot_roc_curve(true_labels_binary, predictions_binary, lb.classes_)

        return df_report

    def plot_roc_curve(self, y_true: List[int], y_pred: List[int], classes: List[str]) -> None:
        """
        Plot the ROC curve for each class.

        Args:
            y_true: True binary labels for each class
            y_pred: Predicted binary labels for each class
            classes: List of class names
        """
        plt.figure(figsize=(10, 8))

        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for random classifier
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Multi-Class Classification")
        plt.legend(loc="lower right")
        plt.show()

classifier = LLMClassifier()
classifier.classify_document(cleaned_text)