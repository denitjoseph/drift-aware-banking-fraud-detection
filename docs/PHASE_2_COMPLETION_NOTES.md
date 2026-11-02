# Phase 2: Data Processing and Machine Learning Model Development

## 1. Objective

Prepare the banking fraud detection dataset, perform feature processing, train machine learning models, compare their performance, select a classification threshold, and create a reusable prediction pipeline.

## 2. Dataset Preprocessing

- Loaded the Kaggle Credit Card Fraud Detection dataset.
- Original dataset shape: 284,807 rows × 31 columns.
- Checked missing values and duplicate records.
- Missing values: 0.
- Duplicate occurrences: 1,081.
- Removed exact duplicate rows.
- Cleaned dataset shape: 283,726 rows × 31 columns.
- Saved the processed dataset to `data/processed/creditcard_cleaned.csv`.

The original raw dataset was preserved.

## 3. Feature Analysis and Preparation

- Inspected the `Time` and `Amount` features.
- Retained the PCA-transformed features `V1` to `V28`.
- Separated features (`X`) and target (`y`).
- Used an 80/20 stratified train-test split.
- Used `random_state=42` for reproducibility.
- Applied StandardScaler to `Time` and `Amount`.
- Fitted the scaler only on training data to prevent data leakage.

## 4. Machine Learning Models

### Logistic Regression

- Used as the baseline model.
- Applied class weighting to handle class imbalance.
- Achieved:
  - Precision: 0.0562
  - Recall: 0.8737
  - F1 Score: 0.1057
  - PR-AUC: 0.6719

### Random Forest

- Trained using 100 estimators.
- Applied balanced class weights.
- Achieved:
  - Precision: 0.9452
  - Recall: 0.7263
  - F1 Score: 0.8214
  - PR-AUC: 0.8012

### Model Comparison

Compared both models using precision, recall, F1 score, PR-AUC, false positives, and false negatives.

Random Forest provided a stronger balance of precision and recall than the initial Logistic Regression baseline.

## 5. Classification Threshold Tuning

- Explored classification thresholds from 0.10 to 0.50.
- Compared precision, recall, F1 score, false positives, and false negatives.
- Selected 0.40 using validation data during the final evaluation process.
- Stored the selected threshold in `configs/model_config.json`.

## 6. Reusable Machine Learning Pipeline

Created a Scikit-learn pipeline containing:

- ColumnTransformer for feature preprocessing.
- StandardScaler for `Time` and `Amount`.
- Random Forest classifier.
- Model preprocessing and prediction combined in one reusable pipeline.

Saved the final refitted pipeline to:

`models/fraud_detection_pipeline.pkl`

The final artifact was trained using the combined 80% non-test portion of the dataset.

## 7. Final Model Evaluation

Used a separate validation dataset for threshold selection and an untouched test split for final evaluation.

Final holdout evaluation results:

| Metric | Result |
|---|---:|
| Precision | 0.9221 |
| Recall | 0.7474 |
| F1 Score | 0.8256 |
| PR-AUC | 0.8133 |

Confusion matrix:

- True Negatives: 56,645
- False Positives: 6
- False Negatives: 24
- True Positives: 71

Note: These holdout metrics were measured using the model trained on the 64% training subset during final evaluation. The saved final artifact was subsequently refitted using the full 80% non-test portion.

## 8. Prediction Testing

Created `src/models/predict.py` to:

- Load the saved pipeline.
- Read the classification threshold from the model configuration.
- Generate a fraud probability.
- Classify a transaction as legitimate or fraudulent.

Tested using a known fraud example from the dataset:

- Actual class: 1
- Fraud probability: 0.6
- Classification threshold: 0.4
- Prediction: FRAUD

This was an in-dataset demonstration, not a test on a genuinely new transaction.

## 9. Model Configuration

Created `configs/model_config.json` to store:

- Model name.
- Model artifact path.
- Classification threshold.
- Threshold selection metric.
- Threshold selection note.

## 10. Important Project Files

- `src/data/analyze_duplicates.py`
- `src/data/preprocess_data.py`
- `src/features/inspect_features.py`
- `src/features/split_data.py`
- `src/features/scale_features.py`
- `src/models/train_logistic_regression.py`
- `src/models/train_random_forest.py`
- `src/models/compare_models.py`
- `src/models/tune_threshold.py`
- `src/models/train_pipeline.py`
- `src/models/final_evaluation.py`
- `src/models/save_final_model.py`
- `src/models/predict.py`
- `configs/model_config.json`

## 11. Generated Artifacts

The following generated files are kept locally and excluded from Git:

- Raw and processed datasets.
- Saved model `.pkl` files.
- Generated evaluation `.csv` and `.json` reports.

## 12. Phase 2 Outcome

Phase 2 established the data preparation and machine learning foundation of the fraud detection platform.

The project now has a reusable Random Forest pipeline, a configurable classification threshold, model evaluation results, and a working prediction script.

The next phase will focus on packaging the model for application serving and preparing the local deployment environment.