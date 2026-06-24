# 🧠 Stroke Prediction Machine Learning Project

## 📌 Problem Statement
Stroke is a medical emergency where the blood supply to part of the brain is cut off. Because outcomes depend heavily on quick recognition, there is immense clinical value in identifying high-risk patients *before* a stroke occurs using routinely collected data (age, lifestyle, pre-existing conditions, etc.). 

**Objective:** Build and compare classification models to predict whether a patient is likely to experience a stroke (`stroke` = 1). 
**Key Metric:** A useful screening model must correctly flag as many true stroke cases as possible. Therefore, **Recall** is prioritized over accuracy, as missing an at-risk patient (false negative) is far more costly than an unnecessary follow-up check (false positive).

## 📊 Dataset Overview
The dataset (`healthcare-dataset-stroke-data.csv`) contains 5,110 patient records.
* **Class Imbalance:** The dataset is highly imbalanced, with only 249 (4.9%) actual stroke cases compared to 4,861 (95.1%) non-stroke cases.
* **Key Features:** Age, average glucose level, hypertension, and heart disease were identified during exploratory data analysis as the strongest predictors of stroke risk. 
* **Data Cleaning:** Missing BMI values (~4% of the data) were imputed using the median to handle the right-skewed distribution robustly.

## 🛠️ Tech Stack & Pipeline
* **Language:** Python
* **Libraries:** `pandas`, `NumPy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Preprocessing Pipeline:** * `StandardScaler` for numeric features (age, glucose level, bmi).
  * `OneHotEncoder` (dropping the first level) for categorical features.
  * `ColumnTransformer` to bundle preprocessing steps cleanly.

## 📈 Modeling & Evaluation
To combat the severe class imbalance, `class_weight='balanced'` was utilized during model training. A `StratifiedKFold` (5 splits) cross-validation strategy was implemented to preserve the 4.9% stroke rate across all folds.

### Model 1: Logistic Regression
* **Recall (Class 1):** 0.88 
* **Accuracy:** 0.73
* **Notes:** Used `GridSearchCV` to optimize the model (Best parameters: `C=1`, `penalty='l2'`, `solver='liblinear'`). Highly interpretable, but produced more false positives.

### Model 2: Decision Tree
* **Notes:** Built a constrained baseline tree (`max_depth=3`, `min_samples_split=20`, `min_samples_leaf=10`) to prevent overfitting on the minority class and provide a clear visual decision path. 

### Model 3: Random Forest (Ensemble)
* **Overview:** Implemented an ensemble of decision trees to capture complex, non-linear relationships in the patient data.
* **Performance:** Successfully reduced the variance and overfitting typically seen in standalone Decision Trees, providing a robust and stable balance between catching actual stroke cases (Recall) and overall Accuracy.
