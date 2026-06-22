# Credit Risk Default Prediction

An end-to-end Machine Learning project for predicting whether a loan applicant is likely to default within the next 12 months. The project combines data preprocessing, feature engineering, model development, hyperparameter tuning, explainability, and an interactive dashboard for model insights.

## Project Overview

This project aims to identify high-risk loan applicants using historical credit and repayment data. Multiple machine learning models were evaluated and compared using industry-standard metrics, with CatBoost selected as the final model.

### Key Features

* Data preprocessing and cleaning
* Advanced feature engineering
* Class imbalance handling using model-level weighting
* Multiple model comparison
* Hyperparameter tuning with Optuna
* SHAP-based model explainability
* Risk segmentation and score band analysis
* Interactive dashboard for predictions and insights

---

## Project Structure

```text
PROJECT2/
│
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks used for analysis and modeling
├── reports/                 # Project reports and documentation
│
├── best_cat.pkl             # Trained CatBoost model
├── dashboard.py             # Interactive dashboard application
├── metrics.csv              # Model evaluation metrics
├── submission.csv           # Test predictions
│
├── X_train.pkl              # Training features
├── X_test.pkl               # Test features
├── y_train.pkl              # Training labels
├── y_test.pkl               # Test labels
│
├── requirements.txt         # Project dependencies
└── README.md
```

---

## Dataset

The project uses multiple datasets containing:

* Applicant Information
* Credit Bureau Records
* Previous Loan History
* Payment Behavior
* Credit Card Usage

These datasets were merged and transformed into a unified analytical dataset for model development.

---

## Feature Engineering

Several categories of engineered features were created:

### Ratio Features

* Debt-to-Income Ratio
* Loan-to-Income Ratio

### Behavioral Features

* Delinquency Rate
* Previous Default Rate
* Payment Consistency

### Credit Utilization Features

* Utilization Ratio
* Credit Used Percentage
* Late Payment Rate

### Time-Based Features

* Average Account Age
* Credit History Length
* Average Days Past Due (DPD)

### Credit Profile Features

* Debt per Account
* Bureau Score Related Metrics

---

## Models Evaluated

The following machine learning algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. CatBoost

### Evaluation Metrics

* ROC-AUC
* Precision
* Recall
* F1 Score
* PR-AUC
* KS Statistic

---

## Final Model Performance

| Model               | ROC-AUC    | PR-AUC     | KS Statistic |
| ------------------- | ---------- | ---------- | ------------ |
| Logistic Regression | 0.8634     | 0.4316     | 0.5948       |
| Decision Tree       | 0.8233     | 0.3052     | 0.5493       |
| Random Forest       | 0.8548     | 0.3880     | 0.5653       |
| XGBoost             | 0.8574     | 0.3898     | 0.5839       |
| LightGBM            | 0.8616     | 0.3981     | 0.5983       |
| CatBoost            | **0.8661** | **0.4297** | **0.5986**   |

### Selected Model

**CatBoost** was selected as the final model due to:

* Highest ROC-AUC score
* Highest KS Statistic
* Strong cross-validation performance
* Competitive PR-AUC
* Excellent interpretability using SHAP

---

## Tech Stack

### Programming Language

* Python 3.14.5

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* CatBoost

### Hyperparameter Optimization

* Optuna

### Model Explainability

* SHAP

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Dashboard

* Streamlit

### Development Environment

* Jupyter Notebook
* VS Code

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd PROJECT2
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Make sure the virtual environment is activated.

Run:

```bash
streamlit run dashboard.py
```

After execution, Streamlit will automatically open a browser window.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## Model Explainability

This project uses SHAP (SHapley Additive exPlanations) to:

* Interpret model predictions
* Identify important risk factors
* Understand feature impact on default probability
* Improve model transparency

---

## Key Insights

* Bureau Score is the strongest predictor of default risk.
* Delinquency-related variables consistently rank among the most important features.
* Higher credit utilization is associated with increased default probability.
* Historical repayment behavior is more predictive than demographic variables.
* Feature engineering significantly improved predictive performance.

---

## Future Improvements

* Deploy the model as a cloud-hosted web application.
* Add real-time prediction capabilities.
* Incorporate additional behavioral and transaction-level features.
* Implement automated model monitoring and retraining pipelines.

---




