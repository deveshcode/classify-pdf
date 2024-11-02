# PDF Classification and Analysis Tool

A comprehensive tool for analyzing, classifying and extracting insights from PDF documents using machine learning and LLM approaches.

## Features

- PDF document processing and text extraction using PyMuPDF
- Machine learning based document classification
- LLM-powered content analysis and insights generation
- Cost estimation for LLM API usage
- Results storage and reporting
- Interactive visualizations using Plotly and Seaborn

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/classify-pdf.git
    ```

2. Install dependencies:
    ```bash
    make install
    ```
3. Set up environment variables: 
    1. Create a `.env` file in the root directory and add the following variables:
        ```bash
        OPENAI_API_KEY=your_openai_api_key
        CLAIM_LOCATION=path_to_data_files
        LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
        ```

4. Run the script:
    ```bash
    make run
    ```

5. To view the dashboard:
    ```bash
    make dash
    ``` 

## Tools used
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-2396F3?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Seaborn](https://img.shields.io/badge/Seaborn-1B998B?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1B998B?style=for-the-badge&logo=pymupdf&logoColor=white)](https://pymupdf.readthedocs.io/en/latest/)
[![Tesseract](https://img.shields.io/badge/Tesseract-1B998B?style=for-the-badge&logo=tesseract&logoColor=white)](https://github.com/madmaze/pytesseract)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Litellm](https://img.shields.io/badge/Litellm-412991?style=for-the-badge&logo=litellm&logoColor=white)](https://litellm.com/)

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

## License
Distributed under the MIT License. See LICENSE for more information.

## Project Team
Name | Contribution %| 
--- |--- | 
Devesh Surve | 100% |


