# kaggle-project-loan-prize
My Kaggle competition solution.
# Loan Prize Prediction – Kaggle ROC-AUC Pipeline

## Object
Build a classification model that predicts the likelihood of a borrower paying back a loan.

## Problem
Financial institutions find it hard to assess loan repayment risk accurately. This is particularly challenging with imbalanced borrower data and a mix of categorical and numerical features.

## How I Solved It
I created a complete Kaggle-ready ML pipeline that:
1. Loads and preprocesses loan data
2. Converts categorical columns to `category` dtype for tree models
3. Trains **XGBoost + LightGBM** using **Stratified K-Fold OOF**
4. Blends predicted probabilities through stacking
5. Evaluates using **ROC-AUC** and RMSE
6. Generates Kaggle submission CSV

## What We Used
- **XGBoost**: This model learns complex non-linear repayment patterns and works well with mixed data.
- **LightGBM**: Offers leaf-wise boosting, faster training time, and effectively manages imbalance.
- **Stacking/Blending**: Combines strengths of the models, improves generalization, and reduces bias.

### Why Use Boosted Tree Models for Loan Prediction?
- No need for extensive feature scaling
- Works effectively with categorical data
- Captures interactions in borrower financial behavior
- High performance in tabular competitions

Using both models provides better stability than depending on one, especially on synthetic Kaggle playground data.

## Tools & Libraries
| Tool | Purpose |
|------|---------|
| Pandas, NumPy | Data cleaning and analysis |
| XGBoost | Gradient Boosted Decision Trees |
| LightGBM | Fast Leaf-wise Boosting |
| Scikit-Learn | Stratified K-Fold, ROC-AUC, Metrics |
| Kaggle Notebook | Model execution and submission |

## Outcomes
- **XGBoost CV AUC:** 0.8712  
- **LightGBM CV AUC:** 0.8698  
- **Blended ROC-AUC:** **0.8754**  
- **Final RMSE (probabilities): 0.3398**
  **REPO STUCTURE**
  
## Submission Format
```csv
id,loan_paid_back
593994,0.85
593995,0.22
593996,0.11
```
