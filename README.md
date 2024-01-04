# st_doc_ext

This repository contains the code for the information extraction app that uses langchain
to extract a structured output from unstructured data for a particular schema.


## Create and activate a venv
```bash
python -m venv <name_of_the_env>
source <name_of_the_env>/bin/activate
```


## Pip install all requirements

```bash
pip install -r requirements.txt
```

## Setup Streamlit Secrets File

This application communicates with the OCR API service to generate the OCR outputs. Spawn the OCR service and then create the secrets.toml file in .streamlit directory at root level and add the following fields to it.

```toml
HOST_URL = ""
OCR_SERVICE_PORT = ""
OCR_PDF_RESP_ENDPOINT = "ocr_pdf"
OCR_IMG_RESP_ENDPOINT = "ocr_image"
OPENAI_API_KEY = ""
```

## Run Streamlit App

To finally run the app:

```bash
streamlit run states.py
```


