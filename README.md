==============================================================
#CUSTOMER CHURN PREDICTION USING MACHINE LEARNING
==============================================================

Predict whether a telecom customer will leave (churn) or stay,
using the Telco Customer Churn dataset. The project covers the
full ML workflow: loading data, exploring it, cleaning,
encoding, training multiple models, evaluating them, finding
the features that drive churn, and predicting for a new
customer.


--------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------
Customer churn is when a customer stops using a company's
service. Winning a new customer usually costs far more than
keeping an existing one, so spotting customers who are likely
to leave is very valuable.

This project builds a binary classifier that predicts churn
(1 = churned, 0 = stayed) and compares three models.


--------------------------------------------------------------
2. DATASET
--------------------------------------------------------------
- Name: Telco Customer Churn (Kaggle / IBM sample data)
- Rows: about 7,000 customers
- Target column: Churn (Yes / No)
- Features include: tenure, contract type, monthly charges,
  total charges, internet service, payment method, tech
  support, streaming services, and more.
- Class balance: only about 26% of customers churn, so the
  data is imbalanced.

The script downloads the dataset automatically from IBM's
public GitHub repository. If the download fails, it asks for
the path to a local copy of the CSV
(WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle also works).


--------------------------------------------------------------
3. PROJECT STRUCTURE
--------------------------------------------------------------
Customer_Churn_Prediction.ipynb   Jupyter Notebook version
Customer_Churn_Prediction.py      Standalone Python script
requirements.txt                  Python dependencies
README.txt                        This file
LICENSE.txt                       License

Files generated after running:
churn_model.pkl                   Best trained model
scaler.pkl                        Fitted StandardScaler
feature_columns.pkl               Column order used in training


--------------------------------------------------------------
###4. TECH STACK
--------------------------------------------------------------
- Python 3.8 or newer
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Joblib


--------------------------------------------------------------
###5. INSTALLATION
--------------------------------------------------------------
1) Clone the repository:

   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>

2) (Recommended) Create a virtual environment:

   python -m venv venv
   source venv/bin/activate        (Windows: venv\Scripts\activate)

3) Install dependencies:

   pip install -r requirements.txt


--------------------------------------------------------------
###6. HOW TO RUN
--------------------------------------------------------------
Option 1: Python script

   python Customer_Churn_Prediction.py

Plots open in separate windows as the script runs. Close each
window to continue.

Option 2: Jupyter Notebook

   jupyter notebook Customer_Churn_Prediction.ipynb

Then run all cells in order.


--------------------------------------------------------------
###7. WORKFLOW
--------------------------------------------------------------
1.  Import libraries
2.  Load the dataset (auto-download with manual fallback)
3.  Explore the data: churn distribution, churn by contract
    type, tenure and monthly charges vs churn
4.  Clean the data:
      - Convert TotalCharges from text to numbers and fill
        gaps with the median
      - Drop customerID (just an identifier)
      - Map Churn from Yes/No to 1/0
5.  Encode categorical columns with one-hot encoding
    (drop_first=True)
6.  Train/test split: 80/20, stratified, random_state=42,
    followed by standard scaling
7.  Train models: Logistic Regression, Random Forest,
    Gradient Boosting
8.  Evaluate: classification report, confusion matrix,
    ROC curves
9.  Feature importance: top 15 features from Random Forest
10. Save the model with Joblib
11. Predict churn probability for a sample customer


--------------------------------------------------------------
###8. MODELS USED
--------------------------------------------------------------
- Logistic Regression: trained on scaled features,
  class_weight="balanced"
- Random Forest: 300 trees, class_weight="balanced"
- Gradient Boosting: default scikit-learn settings

Because only about 26% of customers churn, class_weight=
"balanced" is used (where supported) so the models pay more
attention to the minority class.


--------------------------------------------------------------
###9. EVALUATION METRICS
--------------------------------------------------------------
Models are compared on: Accuracy, Precision, Recall, F1-score
and ROC-AUC.

The best model is selected automatically by ROC-AUC. Recall
and ROC-AUC matter more than raw accuracy here, since a model
that always predicts "stayed" would still look accurate on
imbalanced data.

Run the script to see the exact results on your machine.


--------------------------------------------------------------
###10. KEY INSIGHTS
--------------------------------------------------------------
- Customers on month-to-month contracts are the most likely
  to churn.
- Short-tenure customers churn more than long-term customers.
- High monthly charges are associated with higher churn.


--------------------------------------------------------------
###11. SAVED ARTIFACTS
--------------------------------------------------------------
churn_model.pkl       The best-performing model
scaler.pkl            StandardScaler fitted on training data
feature_columns.pkl   List of feature columns in training order


--------------------------------------------------------------
###12. USING THE SAVED MODEL
--------------------------------------------------------------
   import joblib
   import pandas as pd

   model = joblib.load("churn_model.pkl")
   scaler = joblib.load("scaler.pkl")
   columns = joblib.load("feature_columns.pkl")

   # new_customer must be one-hot encoded the same way as the
   # training data
   new_customer = pd.DataFrame([your_encoded_row], columns=columns)

   # Use scaler.transform(...) only if the best model is
   # Logistic Regression
   prob = model.predict_proba(new_customer)[0][1]
   print(f"Churn probability: {prob:.2%}")

Note: Logistic Regression needs scaled input, while the
tree-based models do not. Check which model was selected as
best before deciding whether to scale.


--------------------------------------------------------------
###13. FUTURE IMPROVEMENTS
--------------------------------------------------------------
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Handle imbalance with SMOTE
- Try XGBoost or LightGBM
- Use a scikit-learn Pipeline to bundle preprocessing and model
- Tune the decision threshold to favour recall
- Build a Streamlit or Flask app to serve predictions


--------------------------------------------------------------
###14. CONTRIBUTING
--------------------------------------------------------------
Contributions are welcome. Open an issue or submit a pull
request:

1. Fork the repo
2. Create a feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -m "Add your feature")
4. Push to the branch (git push origin feature/your-feature)
5. Open a Pull Request


--------------------------------------------------------------
###15. AUTHOR
--------------------------------------------------------------
**Name:** Shashank Deshmukh

**Project:** Customer Churn Prediction

**Purpose:** Machine Learning Project Submission


--------------------------------------------------------------
###16. LICENSE
--------------------------------------------------------------
This project is licensed under the MIT License. See LICENSE.txt
for details.
