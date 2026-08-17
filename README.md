## a. Problem Statement
Credit card default prediction is a critical challenge in the financial services sector. Predicting whether a customer will default on their credit card payment in the upcoming cycle helps financial institutions mitigate credit risk, optimize credit limits, and implement targeted interventions for high-risk accounts. This project develops, evaluates, and deploys multiple machine learning classification models to accurately predict client default status based on demographic data and historical payment records.

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

## c. Github Repository Link
- **GitHub Repository Link:** https://github.com/2025AC05069/ML_Assignment_2
- **Live Streamlit App Link:** https://mlassignment2-ks2lqwcnlhwy75oy69t22m.streamlit.app/#dataset-preview

## d. Models Used

All models were evaluated on the same test dataset using standard scaling. Below is the performance comparison across all mandatory metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| **Decision Tree** | 0.7152 | 0.6079 | 0.3704 | 0.4115 | 0.3899 | 0.2052 |
| **kNN** | 0.7928 | 0.7014 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| **Naive Bayes** | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| **Random Forest (Ensemble)** | 0.8158 | 0.7718 | 0.6550 | 0.3534 | 0.4591 | 0.3848 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves strong overall accuracy (0.8077) and the highest precision (0.6868) among standalone models, but struggles significantly with recall (0.2396), meaning it misses many actual defaulters. |
| **Decision Tree** | Records the lowest overall performance across Accuracy (0.7152), AUC (0.6079), and MCC (0.2052). The model likely overfits the complex features, resulting in poor generalization on test data. |
| **kNN** | Delivers solid baseline accuracy (0.7928) and moderate precision (0.5487), providing a balanced but unremarkable predictive capability across the 23 scaled dimensions. |
| **Naive Bayes** | Yields the highest Recall (0.5539) and the highest F1 Score (0.4975) of all models. It is the best at identifying true defaulters, though it sacrifices precision (0.4515) to do so. |
| **Random Forest (Ensemble)** | Achieves the highest Accuracy (0.8158), highest AUC (0.7718), and highest MCC (0.3848). By utilizing multiple constrained decision trees, it successfully reduces overfitting and provides the most robust overall predictions. |
| **Overall Winner for the dataset?** | **Random Forest (Ensemble)** is the overall winner. It provides the best balance of generalization capability, leading in Accuracy, AUC, and MCC, making it the most reliable model for this dataset. |

