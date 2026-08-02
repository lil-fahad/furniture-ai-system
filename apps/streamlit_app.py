from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("FURNITURE_API_URL", "http://127.0.0.1:8000")
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")

st.set_page_config(page_title="Furniture AI", layout="wide")
st.title("Furniture AI System")
st.caption("One interface for floor-plan analysis, furniture placement, and AI design guidance.")

uploaded = st.file_uploader("Upload a floor plan", type=["png", "jpg", "jpeg", "webp"])
pixels_per_cm = st.number_input("Pixels per centimetre (optional)", min_value=0.0, value=0.0)
use_openai = st.checkbox("Use OpenAI vision refinement")
preferences = st.text_area("Design preferences", placeholder="Modern warm style, neutral colors...")

if uploaded and st.button("Analyze and furnish", type="primary"):
    headers = {"X-API-Key": SERVICE_API_KEY} if SERVICE_API_KEY else {}
    files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    data = {
        "pixels_per_cm": str(pixels_per_cm) if pixels_per_cm > 0 else "",
        "use_openai": str(use_openai).lower(),
        "preferences": preferences,
    }
    with st.spinner("Analyzing..."):
        response = requests.post(
            f"{API_URL}/api/v1/analyze",
            files=files,
            data=data,
            headers=headers,
            timeout=180,
        )
    if response.ok:
        st.success("Analysis complete")
        st.json(response.json())
    else:
        st.error(f"API error {response.status_code}: {response.text}")
