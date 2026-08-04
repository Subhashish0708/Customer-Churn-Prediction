# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer will
**churn** (leave) or **stay**, built on the real **IBM Telco Customer Churn** dataset
(7,043 customers, 21 columns).

## Project structure

```
Customer-Churn-Prediction/
│
├── dataset/
│   └── customer_churn.csv          # Real IBM Telco Customer Churn dataset
│
├── images/                         # Saved charts from the notebook
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_contract.png
│   ├── 03_monthlycharges_vs_churn.png
│   ├── 04_tenure_vs_churn.png
│   ├── 05_service_payment_churn.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_confusion_matrices.png
│   ├── 08_model_comparison.png
│   ├── 09_roc_curve.png
│   └── 10_feature_importance.png
│
├── customer_churn_prediction.ipynb # Full, already-executed notebook
├── README.md
└── requirements.txt
```

## Dataset

The [IBM Telco Customer Churn dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
contains demographic info, account details, and subscribed services for 7,043 customers of a
fictional telecom company, with a `Churn` (Yes/No) label for each.

Key columns: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`,
`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
`TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`,
`PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`.

## Workflow

1. **Data cleaning** — converted `TotalCharges` from text to numeric (11 blank values from
   brand-new customers were filled with 0), dropped the non-predictive `customerID` column.
2. **Exploratory Data Analysis** — churn distribution, churn by contract type, monthly charges
   vs. churn, tenure vs. churn, churn by internet service & payment method, correlation heatmap.
3. **Feature engineering** — label-encoded binary Yes/No columns, one-hot encoded multi-category
   columns (Contract, InternetService, PaymentMethod, etc.), producing 30 model features.
4. **Modeling** — trained baseline **Logistic Regression** and **Random Forest** classifiers on an
   80/20 stratified train/test split.
5. **Hyperparameter tuning** — used 5-fold `GridSearchCV` (optimizing F1-score) to tune both
   models and address class imbalance via `class_weight="balanced"`.
6. **Evaluation** — compared all four model variants on Accuracy, Precision, Recall, F1-score,
   and ROC-AUC; plotted confusion matrices, a metric comparison chart, and ROC curves.
7. **Feature importance** — extracted and visualized the top drivers of churn from the tuned
   Random Forest.
8. **Live prediction** — scored a new, hypothetical customer profile through the full pipeline.

## Key EDA findings

- The dataset is **imbalanced**: ~73% of customers stayed, ~27% churned.
- **Month-to-month contracts churn far more** than one- or two-year contracts.
- **Higher monthly charges** and **fiber-optic internet** are associated with higher churn.
- **Longer tenure** strongly reduces churn — it's one of the most predictive features overall.
- **Electronic check** payers churn more than customers on other payment methods.

## Model results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| Logistic Regression (tuned) | 0.740 | 0.507 | 0.786 | 0.616 | 0.841 |
| Random Forest (baseline) | 0.788 | 0.628 | 0.492 | 0.552 | 0.826 |
| **Random Forest (tuned)** | 0.757 | 0.529 | 0.778 | **0.630** | **0.843** |

*(Best model per metric in bold; see the notebook for the exact numbers and how they were computed.)*

**Takeaway:** Raw accuracy is misleading on this imbalanced dataset. The **tuned Random Forest**
achieves the best F1-score and ROC-AUC — the best overall balance between catching real churners
(recall) and not over-flagging loyal customers (precision). If the business priority is to catch
as many at-risk customers as possible (even at the cost of some false alarms), the tuned models
(with `class_weight="balanced"`) are the better choice, since they roughly **raise recall from
~0.49–0.57 to ~0.78**.

Best hyperparameters found:
- Logistic Regression: `C=1`, `class_weight="balanced"`
- Random Forest: `n_estimators=400`, `max_depth=8`, `min_samples_leaf=1`, `class_weight="balanced"`

## Top churn drivers (Random Forest feature importance)

`tenure`, `MonthlyCharges`, `TotalCharges`, and `Contract_Two year` are the strongest predictors —
confirming the EDA: customers on long contracts with low charges and long tenure are the least
likely to churn.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb
```

Or open the notebook directly in VS Code, JupyterLab, or Google Colab (upload `dataset/customer_churn.csv`
alongside it).

## Next steps for a production system

- Try gradient-boosted models (XGBoost / LightGBM) for a further accuracy boost.
- Use SMOTE or other resampling techniques to address class imbalance more directly.
- Deploy the trained model behind an API and monitor performance drift as customer behavior changes.

## License / attribution

Dataset originally published by IBM for churn-analysis tutorials; used here for educational purposes.
