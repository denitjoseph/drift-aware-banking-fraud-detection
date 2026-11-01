# Dataset Analysis — Banking Fraud Detection

## 1. Dataset Information

- **Dataset:** Credit Card Fraud Detection
- **Source:** Kaggle — mlg-ulb
- **Original file:** `creditcard.csv`
- **Total transactions:** 284,807
- **Total features:** 30
- **Target column:** `Class`

## 2. Dataset Structure

| Column | Description |
|---|---|
| Time | Seconds elapsed since the first transaction |
| V1–V28 | Anonymized PCA-transformed features |
| Amount | Transaction amount |
| Class | Fraud classification target |

## 3. Fraud Distribution

| Transaction Type | Count | Percentage |
|---|---:|---:|
| Legitimate (0) | 284,315 | 99.8273% |
| Fraudulent (1) | 492 | 0.1727% |
| **Total** | **284,807** | **100%** |

## 4. Data Quality Findings

- Missing values: 0
- Duplicate occurrences: 1,081
- Rows involved in duplicate groups: 1,854
- Legitimate rows involved in duplicate groups: 1,822
- Fraudulent rows involved in duplicate groups: 32

## 5. Transaction Amount Statistics

| Statistic | Amount |
|---|---:|
| Mean | 88.35 |
| Standard deviation | 250.12 |
| Minimum | 0.00 |
| Median | 22.00 |
| 75th percentile | 77.165 |
| Maximum | 25,691.16 |

## 6. Initial Observations

1. The dataset has a severe class imbalance.
2. Fraudulent transactions represent only 0.1727% of all records.
3. There are no missing values.
4. Duplicate records require further investigation before cleaning.
5. The V1–V28 features are anonymized and PCA-transformed.

## 7. Planned Data Preparation

- Preserve the original raw dataset.
- Investigate duplicate records before deciding how to handle them.
- Use separate training and testing datasets.
- Evaluate using precision, recall, F1-score and PR-AUC in addition to accuracy.

**Note:** This is an anonymized benchmark dataset, not real production banking transaction data.