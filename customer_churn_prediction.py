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

