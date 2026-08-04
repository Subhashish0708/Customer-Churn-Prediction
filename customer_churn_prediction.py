import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

pd.set_option("display.max_columns", None)

df = pd.read_csv("dataset/customer_churn.csv")                           //Load the dataset
print("Shape:", df.shape)
df.head()

df.info()                                                               //Understand the dataset
df.describe(include="all").T
 
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")                //Data cleaning

print("Missing values created by conversion:", df["TotalCharges"].isnull().sum())
df[df["TotalCharges"].isnull()][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]] 

df["TotalCharges"] = df["TotalCharges"].fillna(0)                    //these are costomer
df.isnull().sum().sum()

df = df.drop(columns=["customerID"])                                 //Drop the unique identifier - it has no predictive value
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(df.shape)
df.head()


churn_counts = df["Churn"].value_counts()                                    //Exploratory Data Analysis
print(churn_counts)
print("\nChurn rate: {:.1f}%".format(df["Churn"].mean() * 100))
plt.figure(figsize=(5, 4))
sns.countplot(x="Churn", data=df, palette=["#4C72B0", "#DD8452"])
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Stayed, 1 = Left)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("images/01_churn_distribution.png")
plt.show()


plt.figure(figsize=(6, 4))                                                 //Churn by contract type
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.legend(title="Churn", labels=["Stayed", "Left"])
plt.tight_layout()
plt.savefig("images/02_churn_by_contract.png")
plt.show()

 
plt.figure(figsize=(5, 4))                                               //Monthly charges vs churn
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn (0 = Stayed, 1 = Left)")
plt.ylabel("Monthly Charges ($)")
plt.tight_layout()
plt.savefig("images/03_monthlycharges_vs_churn.png")
plt.show()

  
plt.figure(figsize=(5, 4))                                             //Tenure vs churn
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure (months) vs Churn")
plt.xlabel("Churn (0 = Stayed, 1 = Left)")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.savefig("images/04_tenure_vs_churn.png")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))                             //Churn by internet service and payment method
sns.countplot(x="InternetService", hue="Churn", data=df, ax=axes[0])
axes[0].set_title("Churn by Internet Service")
axes[0].legend(title="Churn", labels=["Stayed", "Left"])
sns.countplot(y="PaymentMethod", hue="Churn", data=df, ax=axes[1])
axes[1].set_title("Churn by Payment Method")
axes[1].legend(title="Churn", labels=["Stayed", "Left"])
plt.tight_layout()
plt.savefig("images/05_service_payment_churn.png")
plt.show()


numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]          //Correlation heatmap (numeric features)
plt.figure(figsize=(6, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/06_correlation_heatmap.png")
plt.show()



data = df.copy()                                                                               //Feature engineering & encoding
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    data[col] = data[col].map({"Yes": 1, "No": 0})
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
multi_cat_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
data = pd.get_dummies(data, columns=multi_cat_cols, drop_first=True)
print("Final feature count:", data.shape[1] - 1)
data.head()


X = data.drop(columns=["Churn"])                                              //Train / test split
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)



scaler = StandardScaler()                                                             //Logistic Regression
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log_reg = LogisticRegression(max_iter=2000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=["No Churn", "Churn"]))


rf = RandomForestClassifier(n_estimators=300, random_state=42)                         //Random Forest
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=["No Churn", "Churn"]))



fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))                                        //Confusion matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=axes[0])
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=axes[1])
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig("images/07_confusion_matrices.png")
plt.show()