# NAFDAC Counterfeit Drug Detector

### A Hybrid Lookup + Machine Learning System for Detecting Suspicious NAFDAC Drug Registration Numbers

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

---

## 🚀 Live Demo

**[Open the NAFDAC Counterfeit Drug Detector](https://nafdac-counterfeit-drug-detector.streamlit.app/)**

The application is deployed with Streamlit and provides a user-facing interface for checking NAFDAC registration numbers and identifying potentially suspicious drug products.

---

## 📌 Project Overview

The **NAFDAC Counterfeit Drug Detector** is a hybrid rule-based and machine learning application designed to help identify potentially suspicious pharmaceutical products using their NAFDAC registration numbers.

The project addresses a practical problem in Nigeria: counterfeit and suspicious medicines can pose serious risks to patients. A drug may display a NAFDAC number that looks legitimate but may:

- Not exist in the NAFDAC Greenbook
- Exist but have an inactive registration status
- Be associated with different product information
- Follow the structural pattern of a genuine NAFDAC number without being an actual registered number

To address these scenarios, this project combines:

1. **NAFDAC Greenbook lookup**
2. **Product-information consistency checks**
3. **Machine learning classification**

The system prioritizes official Greenbook evidence whenever a submitted NAFDAC number is found in the reference data. The ML model is used as a secondary analysis layer for numbers that are not found.

---

## 🎯 Project Objective

The objective of this project is to develop an MVP that can classify a submitted pharmaceutical product as:

- **Genuine**
- **Suspicious**

and provide an explanation for the resulting verdict.

The system is designed to distinguish between:

- Active NAFDAC registrations
- Inactive registrations
- Existing registrations with mismatched product information
- NAFDAC numbers not found in the Greenbook
- Invalid NAFDAC inputs

---

## 🧠 Why a Hybrid Approach?

A major design decision in this project was to avoid relying exclusively on machine learning.

A NAFDAC number can have a format that looks legitimate without actually being registered.

Therefore:

> **A valid-looking NAFDAC format does not necessarily mean that the registration is genuine.**

The system first checks the NAFDAC number against the Greenbook reference data.

Only when the number is **not found** does the system use the machine learning model to analyze the structural characteristics of the submitted number.

This produces two complementary verification layers:

### Layer 1 — Greenbook Verification

Uses reference data to establish whether the registration exists and what its registration status is.

### Layer 2 — Machine Learning Analysis

Analyzes NAFDAC number structure for numbers that are not found in the Greenbook.

---

## 🏗️ System Architecture

```text
                         User Input
                             │
                             ▼
                    NAFDAC Number Cleaning
                             │
                             ▼
                    ┌──────────────────┐
                    │ Greenbook Lookup  │
                    └──────────────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                Found                Not Found
                  │                     │
                  ▼                     ▼
        Check Registration       Extract NAFDAC
        Status & Product         Structural Features
        Information                     │
                  │                     ▼
                  │              Random Forest
                  │                Classifier
                  │                     │
                  └──────────┬──────────┘
                             ▼
                       Final Verdict
                   Genuine / Suspicious
                             │
                             ▼
                         Explanation
```

---

## 🔎 Greenbook Lookup

The system uses a cleaned NAFDAC Greenbook reference dataset.

The lookup process first normalizes the submitted NAFDAC number and then checks whether it exists in the reference data.

The lookup function can return the following outcomes:

| Lookup Result | Meaning | Final Verdict |
|---|---|---|
| `FOUND_ACTIVE` | Registration exists and is active | **Genuine** |
| `FOUND_INACTIVE` | Registration exists but is inactive | **Suspicious** |
| `FOUND_BUT_MISMATCH` | NAFDAC number exists but supplied product details do not match | **Suspicious** |
| `FOUND_OTHER_STATUS` | Registration exists with another status | **Suspicious** |
| `NOT_FOUND` | NAFDAC number was not found in the Greenbook | ML analysis |
| `INVALID_INPUT` | Invalid or empty NAFDAC input | **Suspicious** |

This means that simply finding a NAFDAC number in the reference data does not automatically make every submitted product genuine.

---

## 🧪 Product Information Verification

Where additional product information is supplied, the application compares it with the corresponding Greenbook record.

The current verification checks include:

- Product name
- Strength
- Route
- Ingredient name
- Dosage form
- Applicant name
- Expiry date

If the NAFDAC number exists but the supplied information does not match the Greenbook record, the system returns a **Suspicious** verdict and reports the detected mismatches.

---

## 📊 Data Collection

The project uses data scraped from the **NAFDAC Greenbook**.

The scraping process collected:

- **7,511 total records**
- **7,503 drug records**

The source data contains product registration information such as:

- Product name
- NAFDAC number
- Ingredient
- Strength
- Dosage form
- Route
- Applicant
- Product category
- Approval date
- Expiry date
- Registration status

Relevant fields were selected for the detection pipeline.

---

## 🧹 Data Cleaning

NAFDAC registration numbers are standardized before lookup and machine learning.

The cleaning process:

1. Handles missing values
2. Converts values to strings
3. Removes non-alphanumeric characters
4. Converts characters to uppercase

Examples:

```text
04-0858   → 040858
A11-0237  → A110237
```

NAFDAC numbers are deliberately kept as **strings** so that leading zeros are preserved.

This is important because converting NAFDAC numbers to numeric data types can remove leading zeros and cause valid registrations to fail during lookup.

---

## 🧬 Synthetic Fake NAFDAC Data

The Greenbook provides genuine registration records but does not provide a labelled dataset of confirmed counterfeit NAFDAC numbers for supervised classification.

To create an initial supervised learning dataset, synthetic fake NAFDAC numbers were generated from observed NAFDAC number structures while avoiding overlap with genuine Greenbook numbers.

The resulting classification dataset contains:

| Class | Records |
|---|---:|
| Genuine | 5,285 |
| Fake | 2,999 |
| **Total** | **8,284** |

The target variable is:

```text
Genuine = 1
Fake    = 0
```

> **Important:** The synthetic fake records are useful for developing and testing the MVP, but they are not equivalent to confirmed real-world counterfeit samples.

---

## ⚙️ Feature Engineering

The machine learning component focuses on structural characteristics of NAFDAC registration numbers.

Features are extracted from the cleaned NAFDAC number, including structural characteristics such as:

- Number length
- Character composition
- Digit/letter characteristics
- Number structure
- `structure_type`

The `structure_type` feature is categorical and is one-hot encoded before model training.

The final trained model uses:

- **214 features**
- **205 known NAFDAC structure types**

---

## 🤖 Machine Learning Model

The current ML model is a:

### Random Forest Classifier

The dataset is divided into training and testing sets using an **80/20 stratified split**.

The ML workflow is:

```text
Genuine Greenbook Records
          +
Synthetic Fake Records
          │
          ▼
     Feature Engineering
          │
          ▼
   Categorical Encoding
          │
          ▼
     Train/Test Split
          │
          ▼
 Random Forest Classifier
          │
          ▼
      Evaluation
```

---

## 📈 Model Evaluation

The current baseline model achieved an overall accuracy of approximately:

### **82.74% Accuracy**

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Fake | 1.00 | 0.53 | 0.69 |
| Genuine | 0.79 | 1.00 | 0.88 |
| **Macro Average** | **0.89** | **0.76** | **0.78** |
| **Weighted Average** | **0.86** | **0.83** | **0.81** |

### Interpretation

The model performs particularly well at identifying the **Genuine** class, with a recall of **1.00** in the current evaluation.

However, the recall for the **Fake** class is **0.53**, meaning that a significant proportion of synthetic fake examples are still classified as genuine by the ML component.

This is an important limitation and an area for improvement.

For this reason, the project does **not** treat the ML model as the sole authority. The Greenbook lookup remains the primary evidence layer when a registration exists in the reference data.

---

## 🔄 Hybrid Prediction Logic

The final prediction pipeline follows this sequence:

```text
Input NAFDAC Number
        │
        ▼
    Clean Input
        │
        ▼
  Greenbook Lookup
        │
   ┌────┴────┐
   │         │
 Found    Not Found
   │         │
   ▼         ▼
Status &   ML Feature
Details    Extraction
   │         │
   │         ▼
   │       Random
   │       Forest
   │         │
   └────┬────┘
        ▼
 Final Verdict
        │
        ▼
 Explanation
```

### Example: Active Registration

```text
Verdict: Genuine

Reason:
NAFDAC number was found and its registration
status is active.
```

### Example: Inactive Registration

```text
Verdict: Suspicious

Reason:
NAFDAC number was found, but its registration
status is inactive.
```

### Example: Product Mismatch

```text
Verdict: Suspicious

Reason:
NAFDAC number was found, but associated product
details do not match the Greenbook record.
```

### Example: Number Not Found

```text
Greenbook Lookup
       ↓
Not Found
       ↓
ML Analysis
       ↓
Genuine / Suspicious
```

---

## 🌐 Streamlit Application

The trained pipeline is integrated into a deployed Streamlit application.

The frontend provides a simple interface through which users can submit NAFDAC registration information and receive:

- Verification result
- Genuine/Suspicious verdict
- Registration status
- Product-detail mismatch information where applicable
- ML-based analysis for NAFDAC numbers not found in the Greenbook

### Live Application

**https://nafdac-counterfeit-drug-detector.streamlit.app/**

---

## 💾 Model Artifacts

The project exports the following artifacts:

```text
artifacts/
├── nafdac_model.joblib
├── nafdac_metadata.joblib
└── greenbook_lookup.csv
```

### `nafdac_model.joblib`

Contains the trained Random Forest classifier.

### `nafdac_metadata.joblib`

Stores model feature metadata, including:

- Feature columns
- Known NAFDAC structure types

### `greenbook_lookup.csv`

Contains the Greenbook reference records used by the lookup system.

The exported reference data contains **7,503 Greenbook records**.

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Collection

- Requests
- BeautifulSoup

### Data Processing

- Pandas
- NumPy
- Regular Expressions

### Machine Learning

- Scikit-learn
- Random Forest Classifier

### Model Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### Model Persistence

- Joblib

### Frontend / Deployment

- Streamlit
- Streamlit Cloud

### Development Environment

- Google Colab
- Jupyter Notebook

### Visualization

- Matplotlib
- Seaborn

---

## 📁 Project Structure

A recommended repository structure is:

```text
NAFDAC-Drug-Detector/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── nafdac_drug_detector.ipynb
│
├── data/
│   └── README.md
│
├── artifacts/
│   ├── nafdac_model.joblib
│   ├── nafdac_metadata.joblib
│   └── greenbook_lookup.csv
│
├── src/
│   ├── preprocessing.py
│   ├── lookup.py
│   ├── feature_engineering.py
│   └── prediction.py
│
└── app/
    └── app.py
```

Adjust the structure above to match the actual files committed to the repository.

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd NAFDAC-Drug-Detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the core dependencies:

```bash
pip install pandas numpy scikit-learn requests beautifulsoup4 matplotlib seaborn joblib streamlit
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application can then be accessed through the local Streamlit URL displayed in the terminal.

---

## 📓 Notebook

The main notebook documents the development process, including:

1. Data collection
2. Data exploration
3. Data cleaning
4. NAFDAC number normalization
5. Greenbook lookup development
6. Synthetic fake data generation
7. Feature engineering
8. Model training
9. Model evaluation
10. Model artifact export
11. Hybrid prediction logic

---

## 🔐 Data and Privacy

The project uses NAFDAC Greenbook registration information for research and development purposes.

The application should not collect or retain unnecessary personal information from users.

---

## ⚠️ Limitations

This project is an **MVP/prototype** and should not be treated as a regulatory or medical verification system.

### Current Limitations

1. The counterfeit examples used for supervised machine learning are synthetic rather than confirmed counterfeit products. Therefore, the ML model should be interpreted as a screening tool rather than a definitive counterfeit detector.

2. The current ML model has a Fake-class recall of 53%, meaning that some suspicious synthetic examples may not be correctly identified by the model. This indicates that further data collection and model improvement are required.

3. The NAFDAC Greenbook data used by the system represents a dataset snapshot. Registration records, product status, approval information, and other regulatory details may change over time, so the reference data requires periodic updating.

4. Verification is not based on the NAFDAC registration number alone. Where available, the system also evaluates product information such as product name, form name, route, strength, applicant name, ingredient name and expiry date against the corresponding Greenbook record.

5. Product-detail consistency checks are dependent on the accuracy and completeness of the information supplied by the user. Incorrect, incomplete, or differently formatted product information may affect the verification result.

6. The hybrid system combines Greenbook lookup, registration-status checks, product-detail consistency checks, and machine learning analysis. However, an ML prediction or automated system verdict cannot independently establish that a physical medicine is definitively genuine or counterfeit.

7. The current MVP does not comprehensively verify physical characteristics of the medicine, such as packaging quality, security features, batch numbers, holograms, tamper evidence, or laboratory composition.

8. Additional regulatory and physical verification may therefore be required for definitive authentication, particularly where a product is flagged as suspicious.

Therefore, the application should be considered a **screening and decision-support tool**, not a replacement for official regulatory verification.

---

## 🔮 Future Improvements

Future versions of the project can include:

- [ ] Improve Fake-class recall
- [ ] Generate more diverse synthetic counterfeit examples
- [ ] Incorporate confirmed real-world counterfeit samples where available
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Threshold optimization
- [ ] Improve ML explanation and confidence reporting
- [ ] Add NAFDAC number image/OCR extraction
- [ ] Add barcode/QR-code scanning
- [ ] Add batch-number verification
- [ ] Add automated Greenbook data updates
- [ ] Improve model monitoring and drift detection
- [ ] Expand the Streamlit application with additional product verification features
- [ ] Develop a mobile-friendly version

---

## 🎥 Project Demonstration

A short demonstration video can be found here:


[[(https://drive.google.com/file/d/1TMy7B9494CMHTfSQ39m_CJj_pg6uWpDV/view?usp=drive_link)]
```

The demonstration should show:

1. Opening the deployed application
2. Entering a NAFDAC number
3. Performing the verification
4. Displaying the verdict
5. Showing the explanation
6. Demonstrating an active/inactive or mismatch case
7. Demonstrating an NAFDAC number that is not found in the Greenbook

---

## 📌 Key Takeaway

The key idea behind the project is:

> **Do not assume that a NAFDAC number is genuine simply because it follows a valid-looking format.**

The NAFDAC Counterfeit Drug Detector combines **official reference-data verification** with **machine learning-based structural analysis** to provide a more practical screening approach.

```text
             Greenbook Evidence
                    +
            Product Verification
                    +
             ML Analysis
                    │
                    ▼
          Hybrid Drug Detector
                    │
                    ▼
         Genuine / Suspicious
                    +
                Explanation
```

---

## ⚖️ Disclaimer

This project is an educational and research-oriented prototype.

A prediction of **Genuine** or **Suspicious** should not be interpreted as definitive proof that a physical medicine is authentic or counterfeit.

For regulatory, medical, pharmaceutical, or safety-critical decisions, users should rely on official NAFDAC verification channels and appropriate professional or regulatory procedures.

---

## 👩🏽‍💻 Author

**Kehinde Balogun**

Data Science & Machine Learning Project

---

## ⭐ Acknowledgements

Data used in this project was obtained from the **NAFDAC Greenbook**.

This project demonstrates how data collection, data preprocessing, rule-based verification, feature engineering, machine learning, and application deployment can be combined to build a practical pharmaceutical product screening system.

