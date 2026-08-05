from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("FURNITURE_API_URL", "http://127.0.0.1:8000")
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")

st.set_page_config(page_title="Furniture AI", layout="wide")
st.title("Furniture AI System")
st.caption("Floor-plan design and trained supplier recommendations in one interface.")

design_tab, supplier_tab = st.tabs(["Design", "Supplier ranker"])

with design_tab:
    uploaded = st.file_uploader("Upload a floor plan", type=["png", "jpg", "jpeg", "webp"])
    pixels_per_cm = st.number_input("Pixels per centimetre (optional)", min_value=0.0)
    use_openai = st.checkbox("Use OpenAI vision refinement")
    preferences = st.text_area(
        "Design preferences", placeholder="Modern warm style, neutral colors..."
    )

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

with supplier_tab:
    category = st.text_input("Furniture category", placeholder="Sofas, lighting, office...")
    col1, col2, col3 = st.columns(3)
    with col1:
        requires_dropshipping = st.checkbox("Dropshipping required")
        max_lead_days = st.number_input("Maximum lead days", min_value=0.0)
    with col2:
        requires_3d_models = st.checkbox("3D models required")
        max_moq = st.number_input("Maximum MOQ", min_value=0.0)
    with col3:
        requires_direct = st.checkbox("Direct fulfillment required")
        max_price = st.number_input("Maximum average price (USD)", min_value=0.0)
    top_k = st.slider("Number of suppliers", min_value=1, max_value=20, value=10)

    if st.button("Rank suppliers", type="primary"):
        params = {
            "category": category or None,
            "requires_dropshipping": requires_dropshipping,
            "requires_3d_models": requires_3d_models,
            "requires_direct_fulfillment": requires_direct,
            "max_lead_days": max_lead_days or None,
            "max_moq": max_moq or None,
            "max_price": max_price or None,
            "top_k": top_k,
        }
        response = requests.get(
            f"{API_URL}/api/v1/suppliers/recommend", params=params, timeout=60
        )
        if response.ok:
            st.dataframe(response.json(), use_container_width=True)
        else:
            st.error(f"API error {response.status_code}: {response.text}")
