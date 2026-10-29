"""
Customer Churn Prediction using Machine Learning

Converted from a Jupyter Notebook (.ipynb) to a standalone Python script.
Run with: python Customer_Churn_Prediction.py
Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, joblib
"""

# Note: This script runs in a normal Python environment; it does not require Google Colab.


# # Customer Churn Prediction using Machine Learning
#
# **Goal:** Predict whether a telecom customer will leave (churn) or stay, using the Telco Customer Churn dataset (Kaggle / IBM).
#
# **Steps:** Load data → Explore → Clean → Encode → Train models → Evaluate → Feature importance → Predict for a new customer


# ## 1. Import libraries


# --- Notebook code cell 2 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ## 2. Load the dataset
#
# This tries to download the dataset automatically. If that fails, it asks you to upload the CSV manually.
# (Kaggle dataset name: **Telco Customer Churn** by blastchar, file `WA_Fn-UseC_-Telco-Customer-Churn.csv`.)


# --- Notebook code cell 4 ---
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(url)
    print("Loaded from URL")
except Exception as e:
    print("Auto-download failed:", e)
    csv_path = input("Enter the path to the Telco Customer Churn CSV file: ").strip().strip('"')
    if not csv_path:
        raise RuntimeError("No CSV path supplied. Download the Telco Customer Churn CSV and run again.")
    df = pd.read_csv(csv_path)

print(df.shape)
print(df.head())


# ## 3. Explore the data


# --- Notebook code cell 6 ---
df.info()


# --- Notebook code cell 7 ---
print("Missing values:\n", df.isnull().sum().sum())
print("\nChurn counts:")
print(df["Churn"].value_counts())
print("\nChurn %:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))


# --- Notebook code cell 8 ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sns.countplot(data=df, x="Churn", ax=axes[0, 0])
axes[0, 0].set_title("Churn distribution")

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 1])
axes[0, 1].set_title("Churn by contract type")

sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Tenure vs churn")

sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Monthly charges vs churn")

plt.tight_layout()
plt.show()


# ## 4. Clean the data
#
# - `TotalCharges` is stored as text (blank for brand-new customers) → convert to numbers
# - `customerID` is just an identifier → drop it


# --- Notebook code cell 10 ---
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(df.head())


# ## 5. Encode categorical columns


# --- Notebook code cell 12 ---
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode all text columns
X = pd.get_dummies(X, drop_first=True)

print("Final feature count:", X.shape[1])
print(X.head())


# ## 6. Train / test split and scaling


# --- Notebook code cell 14 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train:", X_train.shape, " Test:", X_test.shape)


# ## 7. Train models
#
# Only ~26% of customers churn, so the data is imbalanced. We use `class_weight="balanced"` to help the models pay attention to churners.


# --- Notebook code cell 16 ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
trained = {}

for name, model in models.items():
    # Logistic Regression needs scaled data; tree models don't
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    trained[name] = (model, pred, proba)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })

results_df = pd.DataFrame(results).set_index("Model").round(3)
print(results_df)


# ## 8. Evaluate the best model


# --- Notebook code cell 18 ---
best_name = results_df["ROC-AUC"].idxmax()
best_model, best_pred, best_proba = trained[best_name]
print("Best model (by ROC-AUC):", best_name)
print()
print(classification_report(y_test, best_pred, target_names=["Stayed", "Churned"]))


# --- Notebook code cell 19 ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
axes[0].set_title(f"Confusion matrix - {best_name}")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# ROC curves for all models
for name, (m, p, pr) in trained.items():
    fpr, tpr, _ = roc_curve(y_test, pr)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, pr):.3f})")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC curves")
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].legend()

plt.tight_layout()
plt.show()


# ## 9. Which features drive churn?


# --- Notebook code cell 21 ---
rf = trained["Random Forest"][0]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 6))
sns.barplot(x=importances.values, y=importances.index, color="steelblue")
plt.title("Top 15 features influencing churn (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ## 10. Save the model


# --- Notebook code cell 23 ---
joblib.dump(best_model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
print("Saved: churn_model.pkl, scaler.pkl, feature_columns.pkl")

# To download them to your computer:
# from google.colab import files
# files.download("churn_model.pkl")


# ## 11. Predict churn for a new customer


# --- Notebook code cell 25 ---
# Pick a customer from the test set and see the prediction
sample = X_test.iloc[[0]]

if best_name == "Logistic Regression":
    sample_in = scaler.transform(sample)
else:
    sample_in = sample

prob = best_model.predict_proba(sample_in)[0][1]
label = "WILL CHURN" if prob >= 0.5 else "WILL STAY"

print(f"Churn probability: {prob:.2%}")
print(f"Prediction: {label}")
print(f"Actual: {'Churned' if y_test.iloc[0] == 1 else 'Stayed'}")


# ## Summary
#
# - Customers on **month-to-month contracts**, with **short tenure** and **high monthly charges**, are the most likely to churn.
# - Compare the models using **Recall** and **ROC-AUC**, not just accuracy, since churners are the minority class.
# - Ideas to extend: hyperparameter tuning (GridSearchCV), SMOTE for imbalance, XGBoost, or a Streamlit app to serve the model.
