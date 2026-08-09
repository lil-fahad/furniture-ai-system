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

MAX_UPLOAD_BYTES = int(os.getenv("FURNITURE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

if uploaded and st.button("Analyze and furnish", type="primary"):
    payload_bytes = uploaded.getvalue()
    if len(payload_bytes) > MAX_UPLOAD_BYTES:
        st.error(
            f"Upload is {len(payload_bytes) / (1024 * 1024):.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB. Please upload a smaller image."
        )
        st.stop()
    headers = {"X-API-Key": SERVICE_API_KEY} if SERVICE_API_KEY else {}
    files = {"image": (uploaded.name, payload_bytes, uploaded.type)}
    data = {
        "pixels_per_cm": str(pixels_per_cm) if pixels_per_cm > 0 else "",
        "use_openai": str(use_openai).lower(),
        "preferences": preferences,
    }
    try:
        with st.spinner("Analyzing..."):
            response = requests.post(
                f"{API_URL}/api/v1/analyze",
                files=files,
                data=data,
                headers=headers,
                timeout=180,
            )
    except requests.Timeout:
        st.error(f"The API at {API_URL} did not respond within 180 seconds. Try again later.")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API_URL}: {exc}")
        st.stop()
    if response.ok:
        st.success("Analysis complete")
        st.json(response.json())
    else:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API error {response.status_code}: {detail}")
