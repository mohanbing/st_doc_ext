# Copyright 2023 Aditya Mohan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
State Machine and transition code
"""
import json
import enum
import time

from transitions import State
from transitions import Machine

import pandas as pd
import streamlit as st

# from decouple import config

from llm import LLM
from utils import generate_hash, get_ocr_response, build_schema, convert_to_csv

HOST_URL = st.secrets["HOST_URL"]
OCR_SERVICE_PORT = st.secrets["OCR_SERVICE_PORT"]
OCR_PDF_RESP_ENDPOINT = st.secrets["OCR_PDF_RESP_ENDPOINT"]
OCR_IMG_RESP_ENDPOINT = st.secrets["OCR_IMG_RESP_ENDPOINT"]


class AvailableDtype(enum.Enum):
    string = "string"
    integer = "integer"


@enum.unique
class AnalysisStage(enum.IntEnum):
    """
    All Possible Stages
    """

    DEFAULT = enum.auto()
    FILE_UPLOAD = enum.auto()
    SINGLE_FILE_SCHEMA_BUILD = enum.auto()
    MULTI_FILE_SCHEMA_BUILD = enum.auto()
    TEXT_ANALYZE = enum.auto()
    LLM_OUTPUT = enum.auto()


class App(Machine):
    """
    State Machine for the Analysis App
    """

    def __init__(self):
        states = [
            State(AnalysisStage.DEFAULT),
            State(AnalysisStage.FILE_UPLOAD),
            State(AnalysisStage.SINGLE_FILE_SCHEMA_BUILD),
            State(AnalysisStage.MULTI_FILE_SCHEMA_BUILD),
            State(AnalysisStage.TEXT_ANALYZE),
            State(AnalysisStage.LLM_OUTPUT),
        ]

        super().__init__(states=states, initial=AnalysisStage.DEFAULT)

        self.add_transition(
            "openai_api_key_presented",
            source=AnalysisStage.DEFAULT,
            dest=AnalysisStage.FILE_UPLOAD,
        )

        self.add_transition(
            "single_file_uploaded",
            source=AnalysisStage.FILE_UPLOAD,
            dest=AnalysisStage.SINGLE_FILE_SCHEMA_BUILD,
        )

        self.add_transition(
            "multi_files_uploaded",
            source=AnalysisStage.FILE_UPLOAD,
            dest=AnalysisStage.MULTI_FILE_SCHEMA_BUILD,
        )

        self.add_transition(
            "analyze_single",
            source=AnalysisStage.SINGLE_FILE_SCHEMA_BUILD,
            dest=AnalysisStage.TEXT_ANALYZE,
        )

        self.add_transition(
            "analyze_multi",
            source=AnalysisStage.MULTI_FILE_SCHEMA_BUILD,
            dest=AnalysisStage.TEXT_ANALYZE,
        )

        self.add_transition(
            "llm_output",
            source=AnalysisStage.TEXT_ANALYZE,
            dest=AnalysisStage.LLM_OUTPUT,
        )

        self.add_transition(
            "reanalyze",
            source=AnalysisStage.LLM_OUTPUT,
            dest=AnalysisStage.TEXT_ANALYZE,
        )

        self.add_transition(
            "analyze_more",
            source=AnalysisStage.LLM_OUTPUT,
            dest=AnalysisStage.DEFAULT,
        )

        self.add_transition(
            "update_api_key",
            source=AnalysisStage.LLM_OUTPUT,
            dest=AnalysisStage.DEFAULT,
        )

    # def openai_api_key_presented(self) -> None:
    #     llm = OpenAI(openai_api_key=st.session_state.openai_api_key)
    #     st.session_state["llm_object"] = llm


def run() -> None:
    """
    Main Driver Code
    """
    st.title("Information Extraction using LLM")
    st.markdown("""---""")

    if "app" not in st.session_state:
        st.session_state["app"] = App()
        st.session_state["openai_api_key"] = ""
        st.session_state["schema_length"] = 1
        st.session_state["tries"] = 0

    if st.session_state.app.state == AnalysisStage.DEFAULT:
        if st.session_state.tries == 0:
            api_key = st.text_input(
                label="OpenAI API Key",
                help="For more information visit: https://openai.com/pricing",
                value=st.session_state.openai_api_key,
            )
        else:
            st.session_state.openai_api_key = ""
            api_key = st.text_input(
                label="OpenAI API Key",
                help="For more information visit: https://openai.com/pricing",
            )

        st.session_state.openai_api_key = api_key

        submit_btn = st.button(
            label="Submit Key",
            on_click=st.session_state.app.openai_api_key_presented,
            use_container_width=True,
        )

        st.markdown("""---""")

        if st.session_state.tries < 5:
            try_for_free = st.button(
                label="Try for free",
                help="Five trial analysis without paying for an OpenAI API Key",
                use_container_width=True,
            )
        else:
            try_for_free = st.button(
                label="Try for free",
                help="Five trial analysis without paying for an OpenAI API Key",
                disabled=True,
                use_container_width=True,
            )

        if try_for_free:
            st.session_state.tries += 1
            if st.session_state.tries == 5:
                st.error("Free Trials Exhausted!!!", icon="🚨")
                time.sleep(3)
                st.rerun()
            st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state.app.openai_api_key_presented()

    if st.session_state.app.state == AnalysisStage.FILE_UPLOAD:
        uploaded_files = []
        if len(uploaded_files) == 0:
            uploaded_files = st.file_uploader(
                "Choose your image or pdf files for analysis",
                accept_multiple_files=True,
                type=["pdf", "jpg", "png"],
            )

        if len(uploaded_files) == 1:
            # print("Here!")
            st.session_state["uploaded_files"] = uploaded_files
            st.session_state.app.single_file_uploaded()
            st.rerun()

        elif len(uploaded_files) > 1:
            # print("Here2!")
            if len(uploaded_files) > 5:
                uploaded_files = []
                st.error(
                    "For now only a maximum of 5 files can be uploaded at once!",
                    icon="❌",
                )
                st.rerun()

            st.session_state["uploaded_files"] = uploaded_files
            st.session_state.app.single_file_uploaded()
            st.rerun()

        st.session_state["files_uploaded"] = uploaded_files

    if st.session_state.app.state == AnalysisStage.SINGLE_FILE_SCHEMA_BUILD:
        st.markdown("## Schema Builder")
        col2, col3 = st.columns(2, gap="medium")
        # with col1:
        #     displayPDF(st.session_state.files_uploaded[0])

        with col2:
            field_values: list[str | None] = [
                None for i in range(st.session_state.schema_length)
            ]
            dtype_values: list[str | None] = [
                None for i in range(st.session_state.schema_length)
            ]
            required_field: list[bool | None] = [
                None for i in range(st.session_state.schema_length)
            ]
            with st.form("schema_form", clear_on_submit=True):
                for i in range(st.session_state.schema_length):
                    field_values[i] = st.text_input("Field", key=str(i))
                    dtype_values[i] = st.selectbox(
                        "dtype",
                        (dtype.value for dtype in AvailableDtype),
                        key=f"dtype_{i}",
                    )
                    required_field[i] = st.checkbox(
                        label="Required?", value=False, key=f"reqd_{i}"
                    )
                    st.divider()

                temp = st.slider(
                    label="Temperature",
                    min_value=0.0,
                    step=0.1,
                    max_value=1.0,
                    value=0.7,
                    help="The temperature parameter adjusts the randomness of the output. Higher values like 0.7 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
                )

                submit = st.form_submit_button(
                    "Analyze Document",
                    use_container_width=True,
                )

                if submit:
                    st.session_state["field_values"] = field_values
                    st.session_state["dtype_values"] = dtype_values
                    st.session_state["required_field"] = required_field
                    st.session_state["temp"] = temp
                    st.session_state.app.analyze_single()
                    # st.rerun()

        with col3:
            add_field = st.button("➕")
            sub_field = st.button("➖")
            if add_field:
                st.session_state.schema_length += 1
                st.rerun()

            if sub_field:
                if st.session_state.schema_length == 1:
                    st.error("Atleast one field is required in schema", icon="🚨")
                    time.sleep(2)
                    st.rerun()

                st.session_state.schema_length -= 1
                st.rerun()

    if st.session_state.app.state == AnalysisStage.TEXT_ANALYZE:
        # st.text("text analyze")
        print(st.session_state.field_values)
        print(st.session_state.dtype_values)

        my_bar = st.progress(0, text="Analysis in progress. Please wait!")
        st.session_state["llm_output"] = []

        with st.container():
            for cnt, uploaded_file in enumerate(st.session_state.uploaded_files):
                if uploaded_file.tell() > 0:
                    uploaded_file.seek(0)

                file_hash = generate_hash(uploaded_file.read())
                uploaded_file.seek(0)

                my_bar.progress(
                    cnt / len(st.session_state.uploaded_files),
                    text=f"Analyzing {uploaded_file.name}",
                )
                byte_type: str = uploaded_file.name.split(".")[-1]
                print(byte_type)

                if byte_type == "pdf":
                    url: str = (
                        HOST_URL + ":" + OCR_SERVICE_PORT + "/" + OCR_PDF_RESP_ENDPOINT
                    )
                    files = [
                        (
                            "file",
                            (
                                uploaded_file.name,
                                uploaded_file,
                                "application/pdf",
                            ),
                        )
                    ]
                else:
                    url = (
                        HOST_URL + ":" + OCR_SERVICE_PORT + "/" + OCR_IMG_RESP_ENDPOINT
                    )
                    files = [
                        (
                            "file",
                            (
                                uploaded_file.name,
                                uploaded_file,
                                f"image/{byte_type}",
                            ),
                        )
                    ]

                payload = {}
                headers = {}

                resp = get_ocr_response(
                    url=url,
                    payload=payload,
                    files=files,
                    headers=headers,
                    file_hash=file_hash,
                )
                if resp.status_code == 200:
                    st.info("Received OCR Output", icon="ℹ️")
                else:
                    st.warning(
                        "Failed to receive OCR output! Skipping this document",
                        icon="⚠️",
                    )
                    continue

                resp_json = json.loads(resp.text)
                # st.write("Extracting information from text")
                schema = build_schema(
                    field_values=st.session_state.field_values,
                    dtype_values=st.session_state.dtype_values,
                    required=st.session_state.required_field,
                )
                with st.spinner("OpenAI API Analyzing Text"):
                    llm_obj = LLM(
                        temperature=st.session_state.temp,
                        openai_api_key=st.session_state.openai_api_key,
                    )
                    output = llm_obj.analyze_text(resp_json["text"], schema=schema)

                st.session_state.llm_output.append(output)
                # st.write("Received LLM Output!")

            my_bar.progress(
                1.0,
                text="Done Analyzing!",
            )
            st.success("Analysis Complete!")
            time.sleep(2)
            st.session_state.app.llm_output()
            st.rerun()

    if st.session_state.app.state == AnalysisStage.LLM_OUTPUT:
        # print(len(st.session_state.llm_output))
        all_files = [
            uploaded_file.name for uploaded_file in st.session_state.uploaded_files
        ]
        st.markdown("## LLM Output")
        tab_list = st.tabs(all_files)

        with st.sidebar:
            st.markdown("## Schema Builder")
            with st.form("schema_form_llm_output"):
                field_values: list[str | None] = [
                    None for i in range(st.session_state.schema_length)
                ]
                dtype_values: list[str | None] = [
                    None for i in range(st.session_state.schema_length)
                ]
                required_field: list[bool | None] = [
                    None for i in range(st.session_state.schema_length)
                ]
                for i in range(st.session_state.schema_length):
                    field_values[i] = st.text_input(
                        "Field",
                        key=f"sidebar_{i}",
                        value=st.session_state.field_values[i]
                        if i < len(st.session_state.field_values)
                        else "",
                    )

                    dtype_to_idx = {
                        dtype: cnt for cnt, dtype in enumerate(AvailableDtype)
                    }
                    dtype_values[i] = st.selectbox(
                        "dtype",
                        (dtype.value for dtype in AvailableDtype),
                        key=f"sidebar_dtype_{i}",
                        index=dtype_to_idx.get(st.session_state.dtype_values[i], 0)
                        if i < len(st.session_state.dtype_values)
                        else 0,
                    )
                    required_field[i] = st.checkbox(
                        label="Required?",
                        value=st.session_state.required_field[i]
                        if i < len(st.session_state.required_field)
                        else False,
                        key=f"sidebar_reqd_{i}",
                    )
                    st.divider()

                temp = st.slider(
                    label="Temperature",
                    min_value=0.0,
                    step=0.1,
                    max_value=1.0,
                    value=st.session_state.temp,
                    key="sidebar_temp",
                    help="The temperature parameter adjusts the randomness of the \
                        output. Higher values like 0.7 will make the output more \
                        random, while lower values like 0.2 will make it \
                        more focused and deterministic.",
                )

                submit = st.form_submit_button(
                    "Re-Analyze Document", use_container_width=True
                )

                if submit:
                    st.session_state["field_values"] = field_values
                    st.session_state["dtype_values"] = dtype_values
                    st.session_state["required_field"] = required_field
                    st.session_state["temp"] = temp
                    st.session_state.app.reanalyze()
                    st.rerun()

            add_field = st.button("➕", key="sidebar_plus")
            sub_field = st.button("➖", key="sidebar_minus")
            if add_field:
                st.session_state.schema_length += 1
                st.rerun()

            if sub_field:
                if st.session_state.schema_length == 1:
                    st.error("Atleast one field is required in schema", icon="🚨")
                    time.sleep(2)
                    st.rerun()

                st.session_state.schema_length -= 1
                st.rerun()

        for cnt, tab in enumerate(tab_list):
            with tab:
                df = pd.DataFrame(st.session_state.llm_output[cnt])
                st.dataframe(df)

                upload_more = st.button(
                    "Analyze more files",
                    key=f"{cnt}_upload_more",
                    use_container_width=True,
                )
                update_key = st.button(
                    "Update OpenAI Key",
                    key=f"{cnt}_update_key",
                    use_container_width=True,
                )

                if upload_more:
                    st.session_state.app.analyze_more()
                    st.rerun()

                if update_key:
                    st.session_state.app.update_api_key()
                    st.rerun()

                csv = convert_to_csv(df)
                st.download_button(
                    label="Export as CSV 💾",
                    data=csv,
                    file_name="result.csv",
                    mime="text/csv",
                    key=f"{cnt}_export",
                    use_container_width=True,
                )

        # st.json(body=st.session_state.llm_output, expanded=True)


if __name__ == "__main__":
    run()
