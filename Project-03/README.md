## Model Limitations

The credit card fraud detection system has the following limitations:

1. **Class Imbalance:** The dataset contains significantly fewer fraudulent transactions than legitimate transactions.
2. **Anonymized Features:** Features V1–V28 are PCA-transformed and do not have directly understandable meanings.
3. **False Positives:** Legitimate transactions may be incorrectly classified as fraudulent.
4. **False Negatives:** Fraudulent transactions may be incorrectly classified as legitimate.
5. **Prediction Uncertainty:** The predicted fraud probability is an estimate, not a guarantee.
6. **Dataset Limitations:** The model was trained on a specific dataset and may not generalize equally well to all real-world transactions.



## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/interncnw/Sohan-grg-ML.git
```

### 2. Navigate to the Project Folder

```bash
cd Project-03-Fraud-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the Dataset

Download the Credit Card Fraud Detection dataset and place `creditcard.csv` inside the `data` folder.

### 7. Train the Model

```bash
python train_model.py
```

This generates the trained model files inside the `models` folder.

### 8. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

### 9. Run Tests

```bash
python -m pytest -v
```

## Requirements

- Python 3.10 or later
- Credit Card Fraud Detection dataset
- Internet connection for initial dependency installation
