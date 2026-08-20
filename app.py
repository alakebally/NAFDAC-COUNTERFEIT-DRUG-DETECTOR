
# Paste your complete Streamlit code here
import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path

# --- Configuration: Point to Google Drive Artifacts --- #
# This path must match where the artifacts were saved on Google Drive
# Assumes Google Drive is mounted at /content/drive
ARTIFACT_DIR = Path("/content/drive/MyDrive/NAFDAC_Drug_Detector/artifacts")

# --- Load Artifacts (Model, Metadata, Lookup Data) --- #
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(ARTIFACT_DIR / "nafdac_model.joblib")
        metadata = joblib.load(ARTIFACT_DIR / "nafdac_metadata.joblib")
        df_greenbook_lookup = pd.read_csv(ARTIFACT_DIR / "greenbook_lookup.csv", dtype=str)

        return model, metadata, df_greenbook_lookup
    except FileNotFoundError:
        st.error(f"Error: Artifacts not found at {ARTIFACT_DIR}. Please ensure the files are saved correctly.")
        st.stop()

model, metadata, df_greenbook_lookup = load_artifacts()

# Extract metadata for feature engineering
feature_columns = metadata["feature_columns"]
all_structure_types = metadata["all_structure_types"]

# Create the nafdac_lookup index from df_greenbook_lookup
# This recreates the lookup structure used in the notebook's `lookup_nafdac` function
df_greenbook_lookup["status_clean"] = df_greenbook_lookup["status"].astype(str).str.strip().str.lower()
nafdac_lookup = (
    df_greenbook_lookup
    .dropna(subset=["clean_nafdac"])
    .drop_duplicates(subset=["clean_nafdac"])
    .set_index("clean_nafdac")
)

# --- Helper Functions (copied from notebook) --- #
def clean_nafdac(nafdac_string):
    if pd.isna(nafdac_string):
        return None
    return re.sub(r'[^a-zA-Z0-9]', '', str(nafdac_string)).upper()

def get_nafdac_structure(value):
    if pd.isna(value):
        return None
    value = str(value)
    structure = []
    for char in value:
        if char.isalpha():
            structure.append("L")
        elif char.isdigit():
            structure.append("D")
    return "".join(structure)

def extract_nafdac_features(df):
    df_features = df.copy()
    df_features['nafdac_length'] = df_features['clean_nafdac'].str.len()
    df_features['num_digits'] = df_features['clean_nafdac'].apply(lambda x: sum(c.isdigit() for c in x) if isinstance(x, str) else 0)
    df_features['num_letters'] = df_features['clean_nafdac'].apply(lambda x: sum(c.isalpha() for c in x) if isinstance(x, str) else 0)
    df_features['digit_ratio'] = df_features['num_digits'] / df_features['nafdac_length']
    df_features['digit_ratio'] = df_features['digit_ratio'].fillna(0)
    df_features['letter_ratio'] = df_features['num_letters'] / df_features['nafdac_length']
    df_features['letter_ratio'] = df_features['letter_ratio'].fillna(0)

    common_prefixes = ['A', 'B', 'C', '0']
    for prefix in common_prefixes:
        df_features[f'has_prefix_{prefix}'] = df_features['clean_nafdac'].str.startswith(prefix, na=False).astype(int)

    df_features['structure_type'] = df_features['clean_nafdac'].apply(get_nafdac_structure)
    return df_features

def lookup_nafdac(nafdac_number, product_name=None, strength=None, route_name=None, ingredient_name=None, form_name=None, applicant_name=None, expiry_date=None):
    cleaned = clean_nafdac(nafdac_number)
    if cleaned is None or cleaned == "":
        return {
            "result": "INVALID_INPUT",
            "message": "No valid NAFDAC number was provided.",
            "record": None,
            "mismatches": []
        }
    if cleaned not in nafdac_lookup.index:
        return {
            "result": "NOT_FOUND",
            "message": "NAFDAC number was not found in the Greenbook reference data.",
            "record": None,
            "mismatches": []
        }

    record = nafdac_lookup.loc[cleaned]
    mismatches = []

    # Convert all inputs to string and lower for comparison, handling None
    def safe_lower(val): return str(val).strip().lower() if val is not None else ''

    if product_name and safe_lower(product_name) != safe_lower(record["product_name"]):
        mismatches.append(f"Product Name mismatch: Provided '{product_name}' vs Greenbook '{record['product_name']}'")
    if strength and safe_lower(strength) != safe_lower(record["strength"]):
        mismatches.append(f"Strength mismatch: Provided '{strength}' vs Greenbook '{record['strength']}'")
    if route_name and safe_lower(route_name) != safe_lower(record["route_name"]):
        mismatches.append(f"Route Name mismatch: Provided '{route_name}' vs Greenbook '{record['route_name']}'")
    if ingredient_name and safe_lower(ingredient_name) != safe_lower(record["ingredient_name"]):
        mismatches.append(f"Ingredient Name mismatch: Provided '{ingredient_name}' vs Greenbook '{record['ingredient_name']}'")
    if form_name and safe_lower(form_name) != safe_lower(record["form_name"]):
        mismatches.append(f"Form Name mismatch: Provided '{form_name}' vs Greenbook '{record['form_name']}'")
    if applicant_name and safe_lower(applicant_name) != safe_lower(record["applicant_name"]):
        mismatches.append(f"Applicant Name mismatch: Provided '{applicant_name}' vs Greenbook '{record['applicant_name']}'")
    if expiry_date and safe_lower(expiry_date) != safe_lower(record["expiry_date"]):
        mismatches.append(f"Expiry Date mismatch: Provided '{expiry_date}' vs Greenbook '{record['expiry_date']}'")

    status = safe_lower(record["status"])

    if mismatches:
        return {
            "result": "FOUND_BUT_MISMATCH",
            "message": "NAFDAC number found but associated details do not match the Greenbook record.",
            "record": record,
            "mismatches": mismatches
        }
    elif status == "active":
        return {
            "result": "FOUND_ACTIVE",
            "message": "NAFDAC number was found and its registration status is active.",
            "record": record,
            "mismatches": mismatches
        }
    elif status == "inactive":
        return {
            "result": "FOUND_INACTIVE",
            "message": "NAFDAC number was found, but its registration status is inactive.",
            "record": record,
            "mismatches": mismatches
        }
    else:
        return {
            "result": "FOUND_OTHER_STATUS",
            "message": f"NAFDAC number was found with registration status: {record['status']}.",
            "record": record,
            "mismatches": mismatches
        }

def predict_nafdac_status_ml(nafdac_input, product_name=None, strength=None, route_name=None, ingredient_name=None, form_name=None, applicant_name=None, expiry_date=None):
    # Step 1: Use the lookup_nafdac function first, passing all available details
    lookup_result = lookup_nafdac(
        nafdac_input,
        product_name=product_name,
        strength=strength,
        route_name=route_name,
        ingredient_name=ingredient_name,
        form_name=form_name,
        applicant_name=applicant_name,
        expiry_date=expiry_date
    )

    if lookup_result['result'] == "FOUND_ACTIVE":
        return {"verdict": "Genuine", "explanation": lookup_result['message']}
    elif lookup_result['result'] == "FOUND_INACTIVE":
        return {"verdict": "Suspicious", "explanation": lookup_result['message']}
    elif lookup_result['result'] == "FOUND_OTHER_STATUS":
        return {"verdict": "Suspicious", "explanation": lookup_result['message']}
    elif lookup_result['result'] == "FOUND_BUT_MISMATCH":
        return {"verdict": "Suspicious", "explanation": f"{lookup_result['message']} Mismatches: {'; '.join(lookup_result['mismatches'])}"}
    elif lookup_result['result'] == "INVALID_INPUT":
        return {"verdict": "Suspicious", "explanation": lookup_result['message']}
    else: # lookup_result['result'] == "NOT_FOUND"
        cleaned_nafdac = clean_nafdac(nafdac_input)

        if not cleaned_nafdac:
             return {"verdict": "Suspicious", "explanation": "Invalid or empty NAFDAC input for ML prediction."}

        single_nafdac_df = pd.DataFrame({'clean_nafdac': [cleaned_nafdac]})
        single_nafdac_features = extract_nafdac_features(single_nafdac_df)

        # Ensure structure_type is categorical with all known categories
        single_nafdac_features['structure_type'] = pd.Categorical(
            single_nafdac_features['structure_type'],
            categories=all_structure_types
        )

        single_nafdac_encoded = pd.get_dummies(single_nafdac_features, columns=['structure_type'], prefix='structure')

        # Align columns with training data
        # X_final needs to be globally available or re-created here from metadata
        # For simplicity, assuming X_final.columns is known from metadata['feature_columns']
        missing_cols = set(feature_columns) - set(single_nafdac_encoded.columns)
        for c in missing_cols:
            single_nafdac_encoded[c] = 0
        single_nafdac_encoded = single_nafdac_encoded[feature_columns] # Ensure column order

        model_raw_prediction = model.predict(single_nafdac_encoded)
        model_proba = model.predict_proba(single_nafdac_encoded)[0]

        if model_raw_prediction[0] == 1: # Predicted as Genuine by the model
            return {"verdict": "Suspicious", "explanation": f"Not found in Greenbook. ML model predicts structural 'Genuine' (Confidence: {model_proba[1]*100:.2f}%). This NAFDAC number is plausible but unregistered, and thus suspicious. Further verification is recommended."}
        else: # Predicted as Fake by the model
            return {"verdict": "Fake", "explanation": f"Not found in Greenbook. ML model predicts 'Fake' (Confidence: {model_proba[0]*100:.2f}%)."}

# --- Streamlit UI --- #
st.set_page_config(page_title="NAFDAC Number Verification", layout="centered")
st.title("NAFDAC Number Drug Verification System")
st.markdown("---")

st.markdown(
    "Enter a NAFDAC number and optional details to verify its status against the Greenbook and a Machine Learning model."
)

with st.form("nafdac_form"):
    st.subheader("Required Field:")
    nafdac_input = st.text_input("NAFDAC Number", placeholder="e.g., A4-0001")

    st.subheader("Optional Details for Enhanced Verification:")
    col1, col2 = st.columns(2)
    with col1:
        product_name = st.text_input("Product Name", placeholder="e.g., Paracetamol Tablet")
        strength = st.text_input("Strength", placeholder="e.g., 500mg")
        route_name = st.text_input("Route Name", placeholder="e.g., Oral")
        ingredient_name = st.text_input("Ingredient Name", placeholder="e.g., Paracetamol")
    with col2:
        form_name = st.text_input("Form Name", placeholder="e.g., Tablet")
        applicant_name = st.text_input("Applicant Name", placeholder="e.g., ABC Pharma Ltd")
        expiry_date = st.text_input("Expiry Date (YYYY-MM-DD)", placeholder="e.g., 2025-12-31")

    submitted = st.form_submit_button("Verify NAFDAC Number")

    if submitted:
        if not nafdac_input:
            st.error("Please enter a NAFDAC Number to proceed.")
        else:
            # Convert empty strings from text_input to None for the prediction function
            def convert_empty_to_none(s): return s if s else None

            result = predict_nafdac_status_ml(
                nafdac_input,
                product_name=convert_empty_to_none(product_name),
                strength=convert_empty_to_none(strength),
                route_name=convert_empty_to_none(route_name),
                ingredient_name=convert_empty_to_none(ingredient_name),
                form_name=convert_empty_to_none(form_name),
                applicant_name=convert_empty_to_none(applicant_name),
                expiry_date=convert_empty_to_none(expiry_date)
            )

            st.markdown("### Verification Result")
            if result["verdict"] == "Genuine":
                st.success(f"**Verdict:** {result['verdict']}")
                st.write(result["explanation"])
            elif result["verdict"] == "Suspicious":
                st.warning(f"**Verdict:** {result['verdict']}")
                st.write(result["explanation"])
            else: # Fake
                st.error(f"**Verdict:** {result['verdict']}")
                st.write(result["explanation"])

st.markdown("--- ")
st.info("**Note:** This system combines a direct Greenbook lookup with a Machine Learning model. If a number is not found in the Greenbook but appears structurally genuine, it will be flagged as 'Suspicious' for further manual verification.")
