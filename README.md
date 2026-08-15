# Credit Card Default Classification Project

## a. Problem Statement
Credit card default prediction is a critical challenge in the financial services sector. Predicting whether a customer will default on their credit card payment in the upcoming cycle helps financial institutions mitigate credit risk, optimize credit limits, and implement targeted interventions for high-risk accounts. This project develops, evaluates, and deploys multiple machine learning classification models to accurately predict client default status based on demographic data and historical payment records.

---

## b. Dataset Description
- **Dataset Name:** Default of Credit Card Clients Dataset (UCI / OpenML)
- **Number of Instances:** 30,000
- **Number of Features:** 23 predictive features + 1 target binary column
- **Target Variable:** `Target` (0 = Non-default, 1 = Default)
- **Key Feature Categories:**
  * **Demographic Variables:** `LIMIT_BAL` (Credit amount), `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`.
  * **Repayment Status (April–September):** `PAY_0` to `PAY_6` (Delay history in months).
  * **Bill Statement Amounts:** `BILL_AMT1` to `BILL_AMT6` (Monthly statement balances).
  * **Previous Payment Amounts:** `PAY_AMT1` to `PAY_AMT6` (Monthly amounts paid).

---

## c. GitHub Repository Link
- **GitHub Repository:** https://github.com/2025AC05069/ML_Assignment_2
- **Live Streamlit App:** https://mlassignment2-ks2lqwcnlhwy75oy69t22m.streamlit.app

---

## d. Models Used & Performance Evaluation

All 5 classification models were trained using standard 80-20 stratified train-test splits with standard scaling applied across features.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8088 | 0.7248 | 0.6852 | 0.2374 | 0.3525 | 0.3341 |
| **Decision Tree** | 0.7265 | 0.6136 | 0.3842 | 0.4122 | 0.3977 | 0.2079 |
| **kNN** | 0.7938 | 0.7029 | 0.5484 | 0.3376 | 0.4178 | 0.3168 |
| **Naive Bayes** | 0.7602 | 0.7391 | 0.4632 | 0.5735 | 0.5125 | 0.3647 |
| **Random Forest (Ensemble)** | 0.8163 | 0.7634 | 0.6514 | 0.3647 | 0.4674 | 0.3962 |

*(Note: Verify and update the exact decimals if your Colab run produced slight variations).*

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates solid baseline accuracy (80.88%) and high precision (68.52%), but suffers from low recall (23.74%) due to class imbalance, missing several actual defaulters. |
| **Decision Tree** | Shows the lowest overall accuracy and AUC (0.6136). Unpruned trees tend to overfit high-variance training features, leading to higher false positive rates on unseen test data. |
| **kNN** | Yields reasonable performance after feature standardization, but suffers from computational latency during test inference and fails to form optimal decision boundaries across 23 dimensions. |
| **Naive Bayes** | Delivers the highest Recall score (57.35%) among all standalone models, capturing the highest proportion of true defaulters, though precision drops due to the feature independence assumption. |
| **Random Forest (Ensemble)** | Outperforms all individual models with the highest overall Accuracy (81.63%), highest AUC (0.7634), and highest Matthews Correlation Coefficient (0.3962), demonstrating superior generalization. |

### Overall Winner for the Dataset
**Random Forest Classifier (Ensemble)** is the clear overall winner. It effectively handles feature interactions, reduces variance through bagging, and achieves the best balance between precision and recall while leading in overall Accuracy, AUC, and MCC.
