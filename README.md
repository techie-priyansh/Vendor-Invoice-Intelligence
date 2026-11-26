# Vendor Invoice Intelligence

A machine learning application for freight cost prediction and vendor invoice risk flagging, built with Python and Streamlit.

## Features

- **Freight Cost Prediction:** Estimates expected freight charges using invoice quantity and dollar value.
- **Invoice Risk Flagging:** Classifies invoices for manual review using invoice and purchase-related features.
- **Interactive Dashboard:** Provides predictions through a Streamlit web interface.

## Tech Stack

Python, Pandas, SQLite, Scikit-learn, Joblib, Streamlit

## Run Locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/techie-priyansh/Vendor-Invoice-Intelligence.git
cd Vendor-Invoice-Intelligence

pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run src/app.py
```

The application uses trained machine learning models stored in the `models/` directory. The SQLite database is used for model training and data processing.

## Models

- Freight Cost Prediction — Regression
- Invoice Risk Classification — Random Forest Classifier

The invoice risk model flags invoices based on rule-derived risk labels and should be used to prioritize manual review, not as a definitive fraud determination.
