import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Remove excessive top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Main title */
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        color: #111827;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }

    .section-description {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }

    /* Dashboard cards */
    .dashboard-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .card-title {
        font-size: 0.82rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .card-value {
        font-size: 1.65rem;
        color: #111827;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }

    .hero h1 {
        color: white;
        font-size: 2rem;
        margin-bottom: 0.35rem;
    }

    .hero p {
        color: #d1d5db;
        font-size: 0.98rem;
        margin-bottom: 0;
    }

    /* Module badge */
    .module-badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* Result boxes */
    .result-high {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        color: #991b1b;
        margin-top: 1rem;
    }

    .result-low {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        color: #065f46;
        margin-top: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.78rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }

    /* Form */
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    /* =========================================================
   STREAMLIT FORM INPUTS
   ========================================================= */

    div[data-testid="stNumberInput"] label {
        color: #1f2937 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #ffffff !important;
        color: #374151 !important;
        border: none !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #f3f4f6 !important;
    }

    div[data-testid="stNumberInput"] small {
        color: #6b7280 !important;
    }
    /* Divider */
    hr {
        border-color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL ASSETS
# ============================================================
@st.cache_resource
def load_assets():
    freight_model = joblib.load("models/freight_model.pkl")
    risk_classifier = joblib.load("models/risk_classifier.pkl")
    scaler = joblib.load("models/scaler.pkl")

    return freight_model, risk_classifier, scaler


try:
    freight_model, risk_classifier, scaler = load_assets()

except Exception as e:
    st.error(
        f"Unable to load model assets: {e}. "
        "Please ensure all model files exist inside the 'models/' folder."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.2rem 0;">
            <div style="font-size:1.35rem;font-weight:700;">
                🧾 Invoice Intelligence
            </div>
            <div style="font-size:0.8rem;color:#9ca3af;">
                ML-powered finance analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Prediction Module")

    selected_module = st.radio(
        "Select module",
        (
            "Freight Cost Prediction",
            "Invoice Risk Flagging"
        ),
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Business Impact")

    st.markdown(
        """
        **Cost Control**  
        Identify potentially inflated freight charges.

        **Audit Efficiency**  
        Prioritize invoices requiring manual review.

        **Vendor Analytics**  
        Establish data-driven freight cost benchmarks.
        """
    )

    st.divider()

    st.caption("Model Version")
    st.caption("Production v1.0")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>Vendor Invoice Intelligence Portal</h1>
        <p>
            Machine-learning powered analytics for freight estimation,
            invoice anomaly detection, and audit prioritization.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODULE 1 — FREIGHT COST PREDICTION
# ============================================================
if selected_module == "Freight Cost Prediction":

    st.markdown(
        '<div class="module-badge">REGRESSION MODEL</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Freight Cost Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Estimate the expected freight cost of a vendor shipment
            using order quantity and invoice value.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Shipment Quantity",
                min_value=1,
                value=1500,
                step=50,
                help="Total number of units in the shipment."
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollar Amount",
                min_value=1.0,
                value=12000.0,
                step=100.0,
                help="Total dollar value of the invoice."
            )

        st.write("")

        submit_freight = st.form_submit_button(
            "Predict Freight Cost",
            type="primary",
            use_container_width=True
        )

    if submit_freight:

        input_df = pd.DataFrame({
            "Quantity": [quantity],
            "Dollars": [dollars]
        })

        predicted_freight = float(
            freight_model.predict(input_df)[0]
        )

        freight_per_unit = predicted_freight / quantity

        st.markdown("### Prediction Results")

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="card-title">Predicted Freight</div>
                    <div class="card-value">
                        ${predicted_freight:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi2:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="card-title">Shipment Quantity</div>
                    <div class="card-value">
                        {quantity:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi3:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="card-title">Freight / Unit</div>
                    <div class="card-value">
                        ${freight_per_unit:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.info(
            f"Expected freight represents approximately "
            f"{(predicted_freight / dollars) * 100:.2f}% "
            f"of the invoice value."
        )


# ============================================================
# MODULE 2 — INVOICE RISK FLAGGING
# ============================================================
elif selected_module == "Invoice Risk Flagging":

    st.markdown(
        '<div class="module-badge">CLASSIFICATION MODEL</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Invoice Risk Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Analyze freight discrepancies and identify invoices
            that may require manual audit review.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("risk_form"):

        col1, col2 = st.columns(2)

        with col1:

            quantity = st.number_input(
                "Shipment Quantity",
                min_value=1,
                value=1500,
                step=50
            )

            dollars = st.number_input(
                "Invoice Total Amount",
                min_value=1.0,
                value=12000.0,
                step=100.0
            )

        with col2:

            billed_freight = st.number_input(
                "Billed Freight Cost",
                min_value=0.0,
                value=450.0,
                step=10.0
            )

        st.write("")

        submit_risk = st.form_submit_button(
            "Analyze Invoice Risk",
            type="primary",
            use_container_width=True
        )

    if submit_risk:

        # ----------------------------------------------------
        # Calculate expected freight
        # ----------------------------------------------------
        freight_input = pd.DataFrame({
            "Quantity": [quantity],
            "Dollars": [dollars]
        })

        expected_freight = float(
            freight_model.predict(freight_input)[0]
        )

        freight_diff = billed_freight - expected_freight

        freight_per_unit = billed_freight / quantity

        # ----------------------------------------------------
        # Classification features
        # ----------------------------------------------------
        input_df = pd.DataFrame({
            "Quantity": [quantity],
            "Dollars": [dollars],
            "freight_per_unit": [freight_per_unit],
            "freight_diff": [freight_diff]
        })

        risk_pred = int(
            risk_classifier.predict(input_df)[0]
        )

        # ----------------------------------------------------
        # Probability handling
        # ----------------------------------------------------
        probabilities = risk_classifier.predict_proba(input_df)[0]

        classes = list(risk_classifier.classes_)

        if 1 in classes:
            high_risk_probability = probabilities[
                classes.index(1)
            ]
        else:
            high_risk_probability = 0.0

        confidence = max(probabilities)

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------
        st.markdown("### Invoice Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Billed Freight",
                f"${billed_freight:,.2f}"
            )

        with col2:
            st.metric(
                "Expected Freight",
                f"${expected_freight:,.2f}"
            )

        with col3:
            st.metric(
                "Freight Difference",
                f"${freight_diff:,.2f}"
            )

        with col4:
            st.metric(
                "Risk Confidence",
                f"{confidence * 100:.1f}%"
            )

        st.divider()

        # ----------------------------------------------------
        # Risk decision
        # ----------------------------------------------------
        if risk_pred == 1:

            st.markdown(
                f"""
                <div class="result-high">
                    <h3 style="margin:0 0 0.35rem 0;">
                        🚨 Manual Review Required
                    </h3>
                    <div>
                        The model classified this invoice as
                        <strong>high risk</strong> with
                        <strong>{high_risk_probability * 100:.1f}%</strong>
                        high-risk probability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if freight_diff > 0:
                st.warning(
                    f"Billed freight is "
                    f"${freight_diff:,.2f} above the predicted baseline."
                )
            else:
                st.warning(
                    "The invoice was flagged by the risk classifier "
                    "despite the freight charge being below the predicted baseline."
                )

        else:

            st.markdown(
                f"""
                <div class="result-low">
                    <h3 style="margin:0 0 0.35rem 0;">
                        ✓ Normal Invoice Profile
                    </h3>
                    <div>
                        The model classified this invoice as
                        <strong>low risk</strong> with
                        <strong>{confidence * 100:.1f}%</strong>
                        classification confidence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "The billed freight is within the model's expected "
                "invoice profile."
            )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        Vendor Invoice Intelligence System &nbsp;•&nbsp;
        Machine Learning Analytics Dashboard
    </div>
    """,
    unsafe_allow_html=True
)