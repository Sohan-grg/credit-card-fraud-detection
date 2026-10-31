## Model Limitations

The credit card fraud detection system has the following limitations:

1. **Class Imbalance:** The dataset contains significantly fewer fraudulent transactions than legitimate transactions.
2. **Anonymized Features:** Features V1–V28 are PCA-transformed and do not have directly understandable meanings.
3. **False Positives:** Legitimate transactions may be incorrectly classified as fraudulent.
4. **False Negatives:** Fraudulent transactions may be incorrectly classified as legitimate.
5. **Prediction Uncertainty:** The predicted fraud probability is an estimate, not a guarantee.
6. **Dataset Limitations:** The model was trained on a specific dataset and may not generalize equally well to all real-world transactions.

**Disclaimer:** This project is intended for educational purposes. Predictions should not be treated as definitive financial decisions.