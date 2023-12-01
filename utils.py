from functools import lru_cache

import base64
import requests

import streamlit as st
import pandas as pd


def displayPDF(file):
    # Opening file from file path
    # with open(file, "rb") as f:
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
    
def convert_to_csv(df: pd.DataFrame):
    return df.to_csv().encode("utf-8")


def build_schema(field_values: list, dtype_values: list, required: list) -> dict:
    """Builds schema

    Args:
        field_values (list): _description_
        dtype_values (list): _description_
        required (list): _description_

    Returns:
        dict: _description_
    """
    res = dict(properties=dict(), required=list())
    for k, v, req in zip(field_values, dtype_values, required):
        res["properties"][k] = dict(type=v)
        if req:
            res["required"].append(k)

    return res


# @lru_cache(maxsize=32)
def get_ocr_response(
    url: str, payload: dict, headers: dict, files: list
) -> requests.Response:
    """Get OCR Output from either cache or the OCR service

    Args:
        url (str): _description_
        payload (dict): _description_
        headers (dict): _description_
        files (list): _description_

    Returns:
        requests.Response: _description_
    """
    resp = requests.request("POST", url, headers=headers, data=payload, files=files)
    return resp
