import streamlit as st
import pandas as pd
import joblib

from config import FREIGHT_MODEL_PATH, RISK_CLASSIFIER_PATH, RISK_FEATURES

# ============================================================
# APP CONSTANTS
# ============================================================
APP_NAME = "Vendor Invoice Intelligence"
APP_VERSION = "v1.0"
RISK_THRESHOLD = 0.50       # default decision threshold used by predict()
BORDERLINE_BAND = 0.30      # probabilities between this and the threshold = "borderline"

MODULE_FREIGHT = "Freight Cost Prediction"
MODULE_RISK = "Invoice Risk Flagging"

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLES
# ============================================================
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #f6f7f9;
    --surface: #ffffff;
    --border: #e4e7ec;
    --text: #101828;
    --text-2: #344054;
    --muted: #667085;
    --subtle: #98a2b3;
    --brand: #1d4ed8;
    --brand-soft: #eff4ff;
    --success: #067647; --success-soft: #ecfdf3; --success-border: #abefc6;
    --danger: #b42318;  --danger-soft: #fef3f2;  --danger-border: #fecdca;
    --warn: #b54708;    --warn-soft: #fffaeb;    --warn-border: #fedf89;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
}

/* ---------- Base ---------- */
.stApp { background: var(--bg); }
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp div {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }

/* Hide Streamlit chrome (keep header so the sidebar toggle still works) */
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1d2939;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #d0d5dd !important; }
section[data-testid="stSidebar"] hr { border-color: #1d2939; }

.sb-brand { padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid #1d2939; margin-bottom: 1.25rem; }
.sb-logo {
    width: 34px; height: 34px; border-radius: 8px; background: var(--brand);
    color: #fff; font-weight: 700; font-size: 0.85rem;
    display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.6rem;
}
.sb-title { font-size: 1.05rem; font-weight: 650; color: #f9fafb; }
.sb-sub { font-size: 0.78rem; color: #98a2b3; margin-top: 0.1rem; }
.sb-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    color: #667085; margin: 1.1rem 0 0.5rem 0;
}
.sb-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.82rem; color: #d0d5dd; padding: 0.4rem 0;
}
.sb-status { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.75rem; color: #75e0a7; }
.sb-dot { width: 6px; height: 6px; border-radius: 50%; background: #17b26a; display: inline-block; }
.sb-text { font-size: 0.82rem; color: #98a2b3; line-height: 1.55; }
.sb-text b { color: #e4e7ec; font-weight: 600; }

/* ---------- Page header ---------- */
.page-header {
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 1rem;
    padding-bottom: 1.25rem; margin-bottom: 1.75rem; border-bottom: 1px solid var(--border);
}
.eyebrow { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--brand); }
.page-title { font-size: 1.75rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; margin: 0.25rem 0; }
.page-sub { color: var(--muted); font-size: 0.95rem; margin: 0; max-width: 720px; }
.chips { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.chip {
    display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.7rem;
    background: var(--surface); border: 1px solid var(--border); border-radius: 999px;
    font-size: 0.78rem; font-weight: 500; color: var(--text-2);
}
.chip-dot { width: 7px; height: 7px; border-radius: 50%; background: #17b26a; display: inline-block; }

/* ---------- Section header ---------- */
.section-head { margin-bottom: 1rem; }
.tag {
    display: inline-block; padding: 0.2rem 0.55rem; border-radius: 6px;
    background: var(--brand-soft); color: var(--brand);
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
}
.section-title { font-size: 1.2rem; font-weight: 650; color: var(--text); margin-top: 0.5rem; }
.section-desc { color: var(--muted); font-size: 0.9rem; margin-top: 0.2rem; }
.subhead {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); margin: 1.75rem 0 0.75rem 0;
}

/* ---------- Forms & inputs ---------- */
div[data-testid="stForm"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.25rem 1.25rem 0.75rem 1.25rem; box-shadow: var(--shadow);
}
.form-group {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); padding-bottom: 0.4rem; margin-bottom: 0.6rem; border-bottom: 1px solid var(--border);
}
div[data-testid="stNumberInput"] label p { color: var(--text-2) !important; font-weight: 500 !important; font-size: 0.88rem !important; }
.stButton > button, .stFormSubmitButton > button { border-radius: 8px; font-weight: 600; min-height: 2.75rem; }

/* ---------- KPI cards ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin: 0.25rem 0 0.5rem 0; }
.kpi {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1rem 1.1rem; box-shadow: var(--shadow);
}
.kpi.accent { border-top: 3px solid var(--brand); }
.kpi-label { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
.kpi-value {
    font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: 0.35rem;
    letter-spacing: -0.01em; font-variant-numeric: tabular-nums;
}
.kpi-sub { font-size: 0.8rem; color: var(--subtle); margin-top: 0.2rem; }

/* ---------- Verdict banner ---------- */
.verdict {
    display: flex; gap: 0.9rem; align-items: flex-start;
    border: 1px solid; border-radius: var(--radius); padding: 1.1rem 1.25rem;
}
.verdict.high { background: var(--danger-soft); border-color: var(--danger-border); }
.verdict.low  { background: var(--success-soft); border-color: var(--success-border); }
.verdict-icon {
    width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-weight: 700; color: #fff;
}
.verdict.high .verdict-icon { background: var(--danger); }
.verdict.low  .verdict-icon { background: var(--success); }
.verdict-title { font-size: 1rem; font-weight: 650; }
.verdict.high .verdict-title { color: var(--danger); }
.verdict.low  .verdict-title { color: var(--success); }
.verdict-body { font-size: 0.9rem; color: var(--text-2); margin-top: 0.2rem; }

/* ---------- Probability meter ---------- */
.meter {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1rem 1.1rem; box-shadow: var(--shadow); margin-top: 1rem;
}
.meter-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.6rem; }
.meter-label { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
.meter-value { font-size: 1.1rem; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
.meter-track { position: relative; height: 10px; background: #eaecf0; border-radius: 999px; }
.meter-fill { height: 100%; border-radius: 999px; }
.meter-marker { position: absolute; top: -4px; width: 2px; height: 18px; background: var(--text); border-radius: 1px; }
.meter-scale { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--subtle); margin-top: 0.45rem; }

/* ---------- Notes ---------- */
.note {
    border: 1px solid var(--border); background: var(--surface); border-radius: 8px;
    padding: 0.7rem 0.9rem; font-size: 0.85rem; color: var(--text-2); margin-top: 0.75rem;
}
.note.warn { background: var(--warn-soft); border-color: var(--warn-border); color: #7a2e0e; }
.note b { font-weight: 600; }

/* ---------- Empty state ---------- */
.empty {
    border: 1px dashed #d0d5dd; border-radius: var(--radius); padding: 2rem;
    text-align: center; color: var(--muted); font-size: 0.9rem; background: transparent; margin-top: 1.5rem;
}

/* ---------- Footer ---------- */
.app-footer {
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
    color: var(--subtle); font-size: 0.78rem;
    border-top: 1px solid var(--border); margin-top: 3rem; padding-top: 1.25rem;
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ============================================================
# UI HELPERS
# ============================================================
def html(markup: str) -> None:
    """Render HTML safely: collapse indentation so Markdown never treats it as a code block."""
    st.markdown(
        " ".join(line.strip() for line in markup.splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


def section_header(tag: str, title: str, description: str) -> None:
    html(f"""
        <div class="section-head">
            <span class="tag">{tag}</span>
            <div class="section-title">{title}</div>
            <div class="section-desc">{description}</div>
        </div>
    """)


def subhead(text: str) -> None:
    html(f'<div class="subhead">{text}</div>')


def form_group(text: str) -> None:
    html(f'<div class="form-group">{text}</div>')


def kpi_cards(items: list[dict]) -> None:
    cards = []
    for item in items:
        accent = " accent" if item.get("accent") else ""
        sub = f'<div class="kpi-sub">{item["sub"]}</div>' if item.get("sub") else ""
        cards.append(
            f'<div class="kpi{accent}">'
            f'<div class="kpi-label">{item["label"]}</div>'
            f'<div class="kpi-value">{item["value"]}</div>{sub}</div>'
        )
    html(f'<div class="kpi-grid">{"".join(cards)}</div>')


def note(text: str, tone: str = "") -> None:
    html(f'<div class="note {tone}">{text}</div>')


def probability_meter(probability: float, threshold: float) -> None:
    pct = max(0.0, min(probability, 1.0)) * 100
    if probability >= threshold:
        color, band = "var(--danger)", "Above threshold"
    elif probability >= BORDERLINE_BAND:
        color, band = "var(--warn)", "Borderline"
    else:
        color, band = "var(--success)", "Low"
    html(f"""
        <div class="meter">
            <div class="meter-head">
                <span class="meter-label">High-risk probability · {band}</span>
                <span class="meter-value">{pct:.1f}%</span>
            </div>
            <div class="meter-track">
                <div class="meter-fill" style="width:{pct:.1f}%; background:{color};"></div>
                <div class="meter-marker" style="left:calc({threshold * 100:.0f}% - 1px);"></div>
            </div>
            <div class="meter-scale">
                <span>0%</span>
                <span>Decision threshold {threshold * 100:.0f}%</span>
                <span>100%</span>
            </div>
        </div>
    """)


def empty_state(text: str) -> None:
    html(f'<div class="empty">{text}</div>')


# ============================================================
# LOAD MODEL ASSETS
# ============================================================
@st.cache_resource
def load_assets():
    freight_model = joblib.load(FREIGHT_MODEL_PATH)

    # The risk training script saves a bundle containing the estimator
    # and the feature-name list, rather than saving the estimator alone.
    risk_bundle = joblib.load(RISK_CLASSIFIER_PATH)
    if not isinstance(risk_bundle, dict) or "model" not in risk_bundle:
        raise ValueError(
            "Risk model file must contain a bundle with a 'model' entry. "
            "Run src/train_risk_classifier.py to regenerate it."
        )

    risk_model = risk_bundle["model"]
    model_feature_names = risk_bundle.get("feature_names", RISK_FEATURES)

    if list(model_feature_names) != list(RISK_FEATURES):
        raise ValueError(
            "Risk model feature order does not match config.RISK_FEATURES. "
            "Retrain the risk model or update config.py to match the saved model."
        )

    return freight_model, risk_model, list(model_feature_names)


try:
    freight_model, risk_classifier, risk_feature_names = load_assets()
except Exception as exc:
    st.error(f"Unable to load model assets: {exc}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    html("""
        <div class="sb-brand">
            <div class="sb-logo">VI</div>
            <div class="sb-title">Invoice Intelligence</div>
            <div class="sb-sub">Finance analytics · ML decision support</div>
        </div>
    """)

    html('<div class="sb-label">Module</div>')
    selected_module = st.radio(
        "Select module",
        (MODULE_FREIGHT, MODULE_RISK),
        label_visibility="collapsed",
    )

    html(f"""
        <div class="sb-label">Model status</div>
        <div class="sb-row">
            <span>Freight regressor</span>
            <span class="sb-status"><span class="sb-dot"></span>Loaded</span>
        </div>
        <div class="sb-row">
            <span>Risk classifier (X1)</span>
            <span class="sb-status"><span class="sb-dot"></span>Loaded</span>
        </div>
        <div class="sb-row">
            <span>Risk features</span>
            <span>{len(risk_feature_names)}</span>
        </div>
    """)

    html("""
        <div class="sb-label">Business impact</div>
        <div class="sb-text">
            <b>Cost control</b> — estimate expected freight charges.<br>
            <b>Audit efficiency</b> — prioritise invoices for manual review.<br>
            <b>Vendor analytics</b> — support data-driven freight benchmarks.
        </div>
    """)


# ============================================================
# PAGE HEADER
# ============================================================
html(f"""
    <div class="page-header">
        <div>
            <div class="eyebrow">Procurement Analytics</div>
            <div class="page-title">{APP_NAME} Portal</div>
            <p class="page-sub">
                Machine-learning freight estimation and invoice risk classification
                to prioritise audit effort.
            </p>
        </div>
        <div class="chips">
            <span class="chip"><span class="chip-dot"></span>Models online</span>
            <span class="chip">{APP_VERSION}</span>
        </div>
    </div>
""")


# ============================================================
# MODULE 1 — FREIGHT COST PREDICTION
# ============================================================
if selected_module == MODULE_FREIGHT:
    section_header(
        "Regression model",
        "Freight Cost Prediction",
        "Estimate expected freight from shipment quantity and invoice value.",
    )

    with st.form("freight_form"):
        form_group("Shipment details")
        col1, col2 = st.columns(2)
        with col1:
            quantity = st.number_input(
                "Shipment quantity (units)", min_value=1, value=1500, step=50,
                help="Total number of units on the invoice.",
            )
        with col2:
            dollars = st.number_input(
                "Invoice amount (USD)", min_value=1.0, value=12000.0, step=100.0,
                format="%.2f", help="Total invoice value before freight.",
            )

        submit_freight = st.form_submit_button(
            "Predict freight cost", type="primary", use_container_width=True
        )

    if submit_freight:
        freight_input = pd.DataFrame({
            "Quantity": [quantity],
            "Dollars": [dollars],
        })
        predicted_freight = float(freight_model.predict(freight_input)[0])
        freight_per_unit = predicted_freight / quantity
        freight_share = predicted_freight / dollars * 100
        value_per_unit = dollars / quantity

        subhead("Prediction results")
        kpi_cards([
            {"label": "Predicted freight", "value": f"${predicted_freight:,.2f}",
             "sub": "Model estimate", "accent": True},
            {"label": "Freight per unit", "value": f"${freight_per_unit:,.4f}",
             "sub": f"Across {quantity:,} units"},
            {"label": "Freight / invoice value", "value": f"{freight_share:.2f}%",
             "sub": f"Of ${dollars:,.2f}"},
            {"label": "Invoice value per unit", "value": f"${value_per_unit:,.2f}",
             "sub": "Input reference"},
        ])

        if predicted_freight < 0:
            note(
                "<b>Check inputs:</b> the model returned a negative freight estimate, "
                "which usually means the inputs are outside the range seen in training.",
                tone="warn",
            )
        else:
            note(
                "Use this estimate as a benchmark: billed freight well above it is a "
                "candidate for vendor follow-up."
            )
    else:
        empty_state("Enter shipment details and run the prediction to see results.")


# ============================================================
# MODULE 2 — INVOICE RISK FLAGGING (X1 FEATURES)
# ============================================================
elif selected_module == MODULE_RISK:
    section_header(
        "Classification model",
        "Invoice Risk Assessment",
        "Score an invoice against its purchase order to decide whether it needs manual review.",
    )

    with st.form("risk_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            form_group("Invoice")
            invoice_quantity = st.number_input(
                "Invoice quantity", min_value=0.0, value=1500.0, step=50.0,
                help="Units billed on the vendor invoice.",
            )
            billed_freight = st.number_input(
                "Billed freight (USD)", min_value=0.0, value=450.0, step=10.0,
                format="%.2f", help="Freight charged on the invoice.",
            )

        with col2:
            form_group("Purchase order")
            total_item_quantity = st.number_input(
                "Total PO item quantity", min_value=0.0, value=1500.0, step=50.0,
                help="Sum of item quantities across the purchase order.",
            )
            total_brands = st.number_input(
                "Distinct brands on PO", min_value=0, value=5, step=1,
            )
            days_po_to_invoice = st.number_input(
                "Days from PO to invoice", min_value=0.0, value=7.0, step=1.0,
            )

        submit_risk = st.form_submit_button(
            "Analyse invoice risk", type="primary", use_container_width=True
        )

    if submit_risk:
        # Construct columns in the exact feature order used during training.
        feature_values = {
            "invoice_quantity": invoice_quantity,
            "Freight": billed_freight,
            "total_brands": total_brands,
            "total_item_quantity": total_item_quantity,
            "days_po_to_invoice": days_po_to_invoice,
        }
        input_df = pd.DataFrame(
            [[feature_values[name] for name in risk_feature_names]],
            columns=risk_feature_names,
        )

        risk_pred = int(risk_classifier.predict(input_df)[0])

        class_probabilities = risk_classifier.predict_proba(input_df)[0]
        classes = list(risk_classifier.classes_)
        high_risk_probability = (
            float(class_probabilities[classes.index(1)])
            if 1 in classes else 0.0
        )
        confidence = float(max(class_probabilities))

        subhead("Assessment")

        if risk_pred == 1:
            html(f"""
                <div class="verdict high">
                    <div class="verdict-icon">!</div>
                    <div>
                        <div class="verdict-title">Manual review recommended</div>
                        <div class="verdict-body">
                            The X1 classifier flagged this invoice with a high-risk probability of
                            <b>{high_risk_probability * 100:.1f}%</b>. Route it to the audit queue.
                        </div>
                    </div>
                </div>
            """)
        else:
            html(f"""
                <div class="verdict low">
                    <div class="verdict-icon">✓</div>
                    <div>
                        <div class="verdict-title">Not flagged</div>
                        <div class="verdict-body">
                            The classifier predicted <b>not flagged</b> with
                            <b>{confidence * 100:.1f}%</b> confidence. Standard processing applies.
                        </div>
                    </div>
                </div>
            """)

        probability_meter(high_risk_probability, RISK_THRESHOLD)

        qty_ratio = (
            f"{invoice_quantity / total_item_quantity * 100:.1f}%"
            if total_item_quantity > 0 else "n/a"
        )
        freight_per_unit = (
            f"${billed_freight / invoice_quantity:,.4f}"
            if invoice_quantity > 0 else "n/a"
        )

        subhead("Invoice profile")
        kpi_cards([
            {"label": "Model confidence", "value": f"{confidence * 100:.1f}%",
             "sub": "For the predicted class", "accent": True},
            {"label": "Invoice / PO quantity", "value": qty_ratio,
             "sub": f"{invoice_quantity:,.0f} of {total_item_quantity:,.0f} units"},
            {"label": "Billed freight", "value": f"${billed_freight:,.2f}",
             "sub": f"{freight_per_unit} per unit"},
            {"label": "PO to invoice", "value": f"{days_po_to_invoice:,.0f} days",
             "sub": f"{total_brands} brand(s) on PO"},
        ])

        # Data checks: informational only, they do not change the model output.
        if total_item_quantity > 0 and invoice_quantity > total_item_quantity:
            note(
                "<b>Data check:</b> invoice quantity exceeds the purchase-order quantity.",
                tone="warn",
            )
        if invoice_quantity == 0 and billed_freight > 0:
            note("<b>Data check:</b> freight is billed on a zero-quantity invoice.", tone="warn")

        with st.expander("Model input"):
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

        st.caption(
            "Model trained in train_risk_classifier.py on the X1 features with rule-generated "
            "labels. Output is decision support, not a final audit determination."
        )
    else:
        empty_state("Enter invoice and purchase-order details, then run the analysis.")


# ============================================================
# FOOTER
# ============================================================
html(f"""
    <div class="app-footer">
        <span>{APP_NAME} · {APP_VERSION}</span>
        <span>Predictions support, and do not replace, audit judgement.</span>
    </div>
""")