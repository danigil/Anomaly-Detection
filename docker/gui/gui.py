import streamlit as st
import requests

url = 'http://127.0.0.1:8000'

with st.form("my_form"):
    distance=st.number_input(label="Input distance value", min_value=0)
    src_bytes = st.number_input(label="Input src_bytes value", min_value=0)
    dest_bytes = st.number_input(label="Input dest_bytes value", min_value=0)

    submitted = st.form_submit_button("Submit")
    if submitted:
        data = {
            "distance": int(distance),
            "src_bytes": int(src_bytes),
            "dest_bytes": int(dest_bytes)
        }

        response = requests.post(url, json=data)
        st.write(response.text)
