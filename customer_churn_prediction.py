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
