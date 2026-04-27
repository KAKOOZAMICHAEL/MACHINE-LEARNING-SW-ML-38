import os`n"""
Streamlit dashboard — Uganda TB Risk Prediction
Connects to the FastAPI backend for live predictions and displays
district-level risk scores with visualisations.
"""
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Uganda TB Risk Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Uganda District-Level TB Risk Dashboard")
st.markdown(
    "Ensemble model: **Isolation Forest (60%) + Kernel Density Estimation (40%)**"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Single District Prediction")

district_name = st.sidebar.text_input("District name (optional)", "")

feature_defaults = {
    "t_tl": 150000.0,
    "working_age_pct": 0.55,
    "under5_pct": 0.18,
    "district_sex_ratio": 0.98,
    "log_area_km2": 7.5,
    "log_pop_density": 5.2,
    "compactness": 0.65,
    "abs_lat": 1.5,
    "facility_count": 25.0,
    "facility_density": 0.0002,
    "log_facility_per_100k": 2.8,
    "is_urban_survey_cluster": 0.0,
    "region_encoded": 3.0,
}

feature_labels = {
    "t_tl": "Total population",
    "working_age_pct": "Working-age %",
    "under5_pct": "Under-5 %",
    "district_sex_ratio": "Sex ratio",
    "log_area_km2": "Log area (km²)",
    "log_pop_density": "Log pop. density",
    "compactness": "Compactness",
    "abs_lat": "Absolute latitude",
    "facility_count": "Facility count",
    "facility_density": "Facility density",
    "log_facility_per_100k": "Log facilities/100k",
    "is_urban_survey_cluster": "Urban cluster (0/1)",
    "region_encoded": "Region code",
}

inputs = {}
for feat, default in feature_defaults.items():
    inputs[feat] = st.sidebar.number_input(
        feature_labels[feat], value=default, format="%.4f"
    )

predict_btn = st.sidebar.button("Predict Risk", type="primary")

# ── Main panel ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Single District Result")

    if predict_btn:
        payload = {**inputs, "district_name": district_name or None}
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            score = data["risk_score"]
            tier  = data["risk_tier"]
            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[tier]

            st.metric("Risk Score", f"{score:.4f}")
            st.markdown(f"**Risk Tier:** {color} {tier}")
            st.markdown(f"- Isolation Forest score: `{data['iso_score']:.4f}`")
            st.markdown(f"- KDE score: `{data['kde_score']:.4f}`")

            # Gauge bar
            fig, ax = plt.subplots(figsize=(4, 0.6))
            bar_color = "#d62728" if tier == "High" else ("#ff7f0e" if tier == "Medium" else "#2ca02c")
            ax.barh(0, score, color=bar_color, height=0.4)
            ax.barh(0, 1 - score, left=score, color="#e0e0e0", height=0.4)
            ax.set_xlim(0, 1)
            ax.axis("off")
            ax.set_title(f"Risk Score: {score:.3f}", fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Start the server with:\n`uvicorn deployment.app:app --reload`")
        except Exception as exc:
            st.error(f"Error: {exc}")
    else:
        st.info("Adjust parameters in the sidebar and click **Predict Risk**.")

with col2:
    st.subheader("Batch Prediction from CSV")
    st.markdown(
        "Upload a CSV with columns matching the 13 feature names. "
        "An optional `district_name` column is supported."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} rows.")

        required = list(feature_defaults.keys())
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            records = df[required + (["district_name"] if "district_name" in df.columns else [])].to_dict(orient="records")
            # Fill missing district_name
            for r in records:
                r.setdefault("district_name", None)

            try:
                resp = requests.post(f"{API_URL}/predict/batch", json=records, timeout=30)
                resp.raise_for_status()
                results = resp.json()

                res_df = pd.DataFrame(results)
                res_df = res_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
                res_df.index += 1

                st.dataframe(
                    res_df.style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
                    use_container_width=True,
                )

                # Bar chart — top 20
                top20 = res_df.head(20)
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                labels = top20["district_name"].fillna(top20.index.astype(str))
                colors = ["#d62728" if t == "High" else ("#ff7f0e" if t == "Medium" else "#2ca02c")
                          for t in top20["risk_tier"]]
                ax2.barh(labels[::-1], top20["risk_score"][::-1], color=colors[::-1])
                ax2.set_xlabel("Risk Score")
                ax2.set_title("Top 20 Highest-Risk Districts")
                ax2.set_xlim(0, 1)
                st.pyplot(fig2, use_container_width=True)
                plt.close()

                csv_out = res_df.to_csv(index=False).encode()
                st.download_button("Download Results CSV", csv_out, "tb_risk_results.csv", "text/csv")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Start the server first.")
            except Exception as exc:
                st.error(f"Batch prediction error: {exc}")

# ── Model info ────────────────────────────────────────────────────────────────
with st.expander("Model Information"):
    st.markdown("""
| Component | Detail |
|-----------|--------|
| **Model type** | Ensemble (unsupervised) |
| **Algorithm A** | Isolation Forest — 500 trees, contamination 0.20 |
| **Algorithm B** | Kernel Density Estimation — Gaussian kernel, bandwidth 1.0 |
| **Ensemble weights** | IsoForest 60% + KDE 40% |
| **Features** | 13 district-level demographic, spatial, and health-facility features |
| **Target** | TB risk score (0 = low risk, 1 = high risk) |
| **Artifact** | `models/best_model.pkl` |
""")

