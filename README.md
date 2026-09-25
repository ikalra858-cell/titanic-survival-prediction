# 🚢 Titanic Survival Prediction

An end-to-end machine learning classification project that predicts whether a passenger survived the Titanic disaster using passenger and travel-related information.

This project was built as a hands-on ML practice project to implement the complete machine learning workflow — from understanding the data and making feature decisions to model evaluation, tuning, and model serialization.

---

## 📌 Project Overview

The objective is to predict the `Survived` outcome for Titanic passengers.

The project focuses not only on model performance, but also on building a clean and reproducible machine learning workflow:

**Data Understanding → EDA → Feature Engineering → Preprocessing → Baseline → Model Comparison → Cross-Validation → Hyperparameter Tuning → Final Model → Serialization**

---

## 🎯 Problem Statement

The Titanic dataset contains information about passengers such as their passenger class, sex, age, family relationships, fare, and embarkation port.

The target variable is:

| Value | Meaning |
|---|---|
| `0` | Did not survive |
| `1` | Survived |

The goal is to build a classification model that can learn patterns from the available passenger information and predict survival for unseen passengers.

---

## 📊 Dataset

This project uses the **Kaggle Titanic dataset**.

The original dataset contains **891 passenger records and 12 columns**.

### Features Used

| Feature | Description |
|---|---|
| `Pclass` | Passenger class |
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation |
| `FamilySize` | Derived family-size feature |

### Feature Decisions

`FamilySize` was created using:

`FamilySize = SibSp + Parch + 1`

The following raw columns were excluded from the initial modeling approach:

- `PassengerId`
- `Name`
- `Ticket`
- `Cabin`

`Name` and `Ticket` contain high-cardinality and irregular information, while `Cabin` contains substantial missingness. These features could be engineered further, but were excluded to keep this warm-up project focused on the core ML workflow.

---

## 🔍 Exploratory Data Analysis

The analysis investigates:

- Target class distribution
- Survival by sex
- Survival by passenger class
- Interaction between sex and passenger class
- Survival across embarkation ports
- Age distribution and missing values
- Fare distribution
- Family-related features
- Missingness and feature suitability

The EDA was used to guide feature selection and preprocessing decisions rather than simply visualizing the dataset.

---

## ⚙️ Machine Learning Workflow

### 1. Data Exploration
Understanding the dataset structure, data types, missing values, duplicates, and feature characteristics.

### 2. Feature Engineering
Created `FamilySize` from `SibSp` and `Parch`.

### 3. Train-Test Split
The dataset was divided into training and test sets using an **80/20 stratified split**.

### 4. Data Preprocessing

Numerical features:

- Median imputation
- Standard scaling

Categorical features:

- Most-frequent imputation
- One-hot encoding

All preprocessing steps were combined with the model using Scikit-learn pipelines to maintain a consistent transformation process and prevent data leakage.

### 5. Baseline
A `DummyClassifier` was used to establish a majority-class baseline.

### 6. Model Training

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

### 7. Evaluation

Models were evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- Five-fold Cross-Validation

### 8. Hyperparameter Tuning

Logistic Regression's `C` parameter was evaluated using five-fold cross-validation.

The tuned configuration was then evaluated separately on the held-out test set.

### 9. Final Model Selection

The original Logistic Regression model with the default `C = 1` was retained as the final model because the tuned configuration did not provide a meaningful improvement over the default configuration.

---

## 📈 Results

### Final Model

**Logistic Regression**

**Test Accuracy: 80.45%**

The final model achieved approximately **80.45% accuracy on the held-out test set**.

The model comparison and detailed evaluation results are available in the project notebook.

---

## 💾 Model Serialization

The complete fitted preprocessing + Logistic Regression pipeline was serialized using **Joblib**.

This allows the trained pipeline to be loaded later and used for predictions without rebuilding the preprocessing steps manually.

Saved model:

```text
models/
└── titanic_logistic_regression_pipeline.joblib
```

---

## 🗂️ Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   └── raw/
│       └── train.csv
│
├── models/
│   └── titanic_logistic_regression_pipeline.joblib
│
├── notebooks/
│   └── titanic_survival_prediction.ipynb
│
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

**Language**
- Python

**Data Analysis**
- Pandas
- NumPy

**Visualization**
- Matplotlib
- Seaborn

**Machine Learning**
- Scikit-learn

**Model Serialization**
- Joblib

**Development**
- Jupyter Notebook
- VS Code
- Git & GitHub

---

## ▶️ How to Access & Run

### 1. Clone the repository

```bash
git clone https://github.com/ikalra858-cell/titanic-survival-prediction.git
```

### 2. Navigate to the project

```bash
cd titanic-survival-prediction
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 4. Open the notebook

Navigate to:

```text
notebooks/titanic_survival_prediction.ipynb
```

and open it using Jupyter Notebook, JupyterLab, or VS Code.

### 5. Access the serialized model

The trained model pipeline is available at:

```text
models/titanic_logistic_regression_pipeline.joblib
```

It can be loaded using Joblib and used for predictions on compatible passenger data.

---

## 📓 Project Notebook

The complete analysis, visualizations, model training, evaluation, tuning, and final conclusions are available in:

**`notebooks/titanic_survival_prediction.ipynb`**

---

## 🚀 Key Takeaways

This project provided hands-on practice with:

- Translating a classification problem into an ML workflow
- Performing exploratory data analysis
- Making feature inclusion/exclusion decisions
- Building preprocessing pipelines
- Comparing multiple classification algorithms
- Using cross-validation
- Performing hyperparameter tuning
- Selecting a final model based on evidence
- Evaluating a model on held-out data
- Serializing and reloading a complete ML pipeline

---

## 📌 Note

This project is intended as a machine learning practice project focused on implementing an end-to-end workflow using a well-known classification dataset.