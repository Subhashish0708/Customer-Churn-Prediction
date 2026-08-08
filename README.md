# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to **churn (leave a service)** or **stay**, based on customer information such as tenure, monthly charges, contract type, internet service, and support calls.

The project uses **Python, Pandas, Scikit-learn, Matplotlib, Seaborn, and Streamlit** to perform data analysis, train machine learning models, evaluate their performance, and provide an interactive prediction interface.

---

## 📌 Project Overview

Customer churn is an important problem for businesses because losing existing customers can directly affect revenue.

This project analyzes customer information and uses machine learning to identify customers who are more likely to leave a service.

The project includes:

* Data loading and preprocessing
* Exploratory Data Analysis (EDA)
* Data visualization
* Feature encoding
* Logistic Regression
* Random Forest Classification
* Model evaluation
* Feature importance analysis
* Model saving using Joblib
* Interactive Streamlit web application

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze customer data.
2. Identify factors associated with customer churn.
3. Build a machine learning model to predict churn.
4. Compare different classification algorithms.
5. Identify important features influencing churn.
6. Create an interactive web application for predictions.

---

## 🛠️ Technologies Used

| Technology       | Purpose                           |
| ---------------- | --------------------------------- |
| Python           | Programming language              |
| Pandas           | Data manipulation                 |
| NumPy            | Numerical computation             |
| Matplotlib       | Data visualization                |
| Seaborn          | Statistical visualization         |
| Scikit-learn     | Machine learning                  |
| Joblib           | Model serialization               |
| Streamlit        | Web application                   |
| Jupyter Notebook | Data analysis and experimentation |

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── dataset/
│   └── customer_churn.csv
│
├── images/
│   └── screenshots
│
├── customer_churn_prediction.ipynb
├── customer_churn_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset contains customer information used to predict whether a customer will churn.

### Features

| Feature         | Description                                          |
| --------------- | ---------------------------------------------------- |
| CustomerID      | Unique customer identifier                           |
| Age             | Age of the customer                                  |
| Tenure          | Number of months the customer has stayed             |
| MonthlyCharges  | Customer's monthly charges                           |
| Contract        | Type of customer contract                            |
| InternetService | Type of internet service                             |
| SupportCalls    | Number of customer support calls                     |
| TotalCharges    | Total amount charged to the customer                 |
| Churn           | Target variable indicating whether the customer left |

### Target Variable

```text
Yes → Customer churned
No  → Customer stayed
```

---

## 🔄 Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Loading
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Data Visualization
       ↓
Feature Encoding
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Feature Importance
       ↓
Churn Prediction
       ↓
Streamlit Web Application
```

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model for predicting whether a customer will churn.

### 2. Random Forest

Random Forest is used as a second classification model. It combines multiple decision trees and can capture relationships between different customer features.

The models are compared based on their prediction performance.

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Example evaluation:

```python
accuracy_score(y_test, y_pred)
```

Classification report:

```python
classification_report(y_test, y_pred)
```

---

## 🔍 Feature Importance

The Random Forest model is also used to identify which customer attributes contribute most to churn prediction.

```python
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})
```

This helps businesses understand which customer characteristics may require attention.

---

## 💾 Model Saving

The trained Random Forest model is saved using Joblib:

```python
joblib.dump(rf_model, "customer_churn_model.pkl")
```

The saved model can then be loaded by the Streamlit application without retraining it.

---

## 🖥️ Streamlit Application

The project includes an interactive web application built with Streamlit.

Users can enter:

* Age
* Tenure
* Monthly Charges
* Contract Type
* Internet Service
* Support Calls
* Total Charges

The application then predicts whether the customer is likely to:

```text
🟢 Stay
```

or

```text
🔴 Churn
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Customer-Churn-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 📸 Application Screenshots

Add screenshots of your application inside the `images` folder and update this section.

### Streamlit Application

![Customer Churn Prediction App](images/app_screenshot.png)

### Data Analysis

![Data Analysis](images/eda_screenshot.png)

### Model Evaluation

![Model Evaluation](images/model_screenshot.png)

---

## 💡 Example Prediction

A customer with:

* Low tenure
* High monthly charges
* Month-to-month contract
* High number of support calls

may have a higher probability of churn.

The application provides a prediction based on the trained machine learning model.

---

## 🔮 Future Improvements

The project can be improved further by:

* Using a larger real-world customer churn dataset
* Adding more customer features
* Implementing additional machine learning algorithms
* Hyperparameter tuning
* Cross-validation
* Probability-based churn prediction
* Adding customer retention recommendations
* Deploying the Streamlit application online
* Adding a database for customer records

---

## 🎓 Learning Outcomes

Through this project, I learned how to:

* Work with structured datasets
* Perform data preprocessing
* Conduct exploratory data analysis
* Create meaningful visualizations
* Encode categorical variables
* Train classification models
* Compare machine learning algorithms
* Evaluate model performance
* Analyze feature importance
* Save and load trained machine learning models
* Build an interactive machine learning web application using Streamlit

---

## 👨‍💻 Author

**Subhashish Santosh Barik**

GitHub: `https://github.com/Subhashish0708`

LinkedIn: `https://www.linkedin.com/in/subhashish-barik-687339276/`

---

## 📄 License

This project is created for **educational and internship purposes**.
