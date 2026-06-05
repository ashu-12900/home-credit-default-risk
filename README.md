## Problem
Home Credit serves unbanked customers who lack traditional 
credit history. The goal is to predict loan default probability 
to reduce financial risk while ensuring creditworthy applicants 
aren't rejected.

## Tech Stack
Python | XGBoost | Optuna | Pandas | Scikit-learn | NumPy

## Home Credit Default Risk

Binary classification to predict loan default probability using the Home Credit Kaggle dataset.

## Results
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.7795 |

## Approach

**Data** — Merged 7 tables (application_train, bureau, previous_application, etc.) via SK_ID_CURR aggregations.

**Feature Engineering**
- Credit utilization & debt ratios
- Age and employment length features
- DPD (days past due) ratio features

**Model** — XGBoost tuned with Optuna (~100 trials), threshold-optimized for F1.

**Pipeline**
- features.pkl saved after preprocessing to prevent train/eval mismatch

## Files
- data_pre.py — merges tables, engineers features, saves features.pkl
- train.py — loads features.pkl, runs Optuna tuning, trains XGBoost
- evaluation_script.py — loads model, evaluates on test set

## How to Run
python data_pre.py
python train.py
python evaluation_script.py

## Dataset
Home Credit Default Risk — Kaggle 2018
https://www.kaggle.com/competitions/home-credit-default-risk/data
Download and place CSVs in the project root.
