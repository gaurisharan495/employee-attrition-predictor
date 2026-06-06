# Employee Attrition Predictor

A machine learning pipeline to predict employee attrition on the 
IBM HR Analytics dataset (1,470 employees, 35 features), comparing 
Logistic Regression, Random Forest, and XGBoost classifiers.

Built during internship at OnActuate Consulting.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 73.8% | 34.0% | 68.1% | 45.4% | 79.96% |
| Random Forest | 84.4% | 51.5% | 36.2% | 42.5% | 78.97% |
| XGBoost | 86.1% | 62.5% | 31.9% | 42.3% | 77.79% |

5-fold Cross-Validated ROC-AUC (Random Forest): **0.8199**

Logistic Regression recommended for deployment — highest Recall (68%)
minimises missed attrition cases, which matters more than raw accuracy
in an HR retention context.

---

## Pipeline

- Drops Employee_ID; target encoded as 0/1
- Numeric features: median imputation
- Categorical features: most-frequent imputation + OneHotEncoding
- Stratified 80/20 train-test split
- Class imbalance handled via class_weight='balanced' and
  scale_pos_weight for XGBoost
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- 5-fold cross-validation on Random Forest

---

## Dataset

IBM HR Analytics Employee Attrition dataset (public, via Kaggle).
1,470 records, 35 features including Age, Job Role, Monthly Income,
Overtime, Years at Company, Work-Life Balance.
Target: Attrition (Yes / No).

---

## How to Run

```bash
pip install -r requirements.txt
# Place IBM HR dataset as dataset.csv in the same folder
jupyter notebook employee_attrition_predictor.ipynb
```
