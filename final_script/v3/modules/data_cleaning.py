import re
from loguru import logger
from modules.log_config import track_time

@track_time
def refined_clean_text(text):
    """
    Clean the extracted text by removing unwanted special characters, symbols, and artifacts.
    """
    logger.info(f"Beginning text cleaning")
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
    logger.info(f"Text cleaning completed")
    return text
