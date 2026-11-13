# Diabetes Prediction using Machine Learning

Predicts whether a patient has diabetes from 8 health measurements, using Python and scikit-learn. The focus is on catching diabetic patients (recall), not just overall accuracy.

## Dataset
Pima Indians Diabetes Dataset: 768 patients, 8 features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and a binary Outcome. About 35% of patients have diabetes.

## Workflow
1. Exploratory data analysis (count plots, boxplots, histograms, correlation heatmap)
2. Data cleaning: found hidden zeros in 5 columns, treated them as missing values and filled them with the median
3. 80/20 stratified train/test split and feature scaling (scaler fitted on training data only, to avoid data leakage)
4. Trained a baseline Logistic Regression model
5. Evaluated with a confusion matrix, precision, recall and F1
6. Improved the model with class weighting and a lower decision threshold, compared with Random Forest, and confirmed with 5-fold cross-validation

## Results (154 test patients)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic (baseline) | 0.708 | 0.600 | 0.500 | 0.545 |
| Logistic (balanced) | 0.734 | 0.603 | 0.704 | 0.650 |
| Logistic (threshold 0.4) | 0.734 | 0.614 | 0.648 | 0.631 |
| Random Forest | 0.740 | 0.652 | 0.556 | 0.600 |

5-fold cross-validation recall: Logistic 0.58, Logistic balanced 0.70, Random Forest 0.58.

## Key findings
- Glucose has the strongest link to diabetes (correlation 0.47), followed by BMI and Age.
- The baseline model had 71% accuracy but caught only half of diabetic patients.
- Class weighting raised recall from 0.50 to 0.70 with accuracy also improving to 0.73.
- Logistic Regression (balanced) was chosen as the best model: it catches the most diabetic patients and is easy to explain.

## Limitations
Small dataset (768 patients), and the model is for learning purposes only. It is not a medical tool.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Google Colab

## Files
diabetes_prediction.ipynb: full notebook with code and outputs

## Author
Nishanth Sivaraman, B.Tech AIML
