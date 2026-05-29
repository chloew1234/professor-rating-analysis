# Professor Rating Analysis
**NYU Principles of Data Science — Capstone Project**

A statistical analysis of 15,000+ RateMyProfessor entries to uncover what factors drive professor ratings — including gender bias, teaching difficulty, online instruction, and "attractiveness" perception.

---

## Overview

This project analyzes a dataset of 89,893 professor records scraped from RateMyProfessor.com. After data cleaning, the working sample contains **20,577 professors** (those with ≥ 6 ratings, to ensure reliable averages).

The analysis covers hypothesis testing, regression modeling, and binary classification — answering 10 research questions about what predicts professor quality ratings.

---

## Key Findings

| Question | Finding |
|---|---|
| Gender bias | Male professors rated **0.056 points higher** than equally experienced females (p < 0.005) |
| Experience | More experienced professors score slightly higher (Spearman ρ = 0.037, p < 0.005) |
| Difficulty | Strong negative relationship — each +1 difficulty point → **−0.74 rating points** (R² = 0.40) |
| Online teaching | Online-heavy professors receive **lower** ratings (Welch t = −4.89, p < 0.005) |
| "Would take again" | Strong positive correlation with rating (β = 0.26, R² = 0.77) |
| "Hot" professors | Rated significantly higher than non-hot professors (p < 0.005) |
| Regression (difficulty only) | R² = 0.407, RMSE = 0.723 |
| Regression (all features) | R² = **0.804**, RMSE = 0.377 — doubles explained variance |
| Classification (rating only) | AUROC = 0.790 predicting "pepper" status |
| Classification (all features) | AUROC = **0.801** — modest improvement with more predictors |
| Geographic extra credit | New England states (NH, VT, MA) rate professors highest; TX, CA, FL rate lowest |

---

## Methods

- **Data cleaning:** Filtered to professors with ≥ 6 ratings; handled missing values via dropna per question
- **Hypothesis testing:** Welch's t-test (unequal variances), Spearman & Pearson correlation
- **Regression:** OLS (statsmodels) and Linear Regression (scikit-learn), 80/20 train-test split
- **Classification:** Logistic Regression with `class_weight='balanced'` to handle class imbalance; evaluated with AUROC, precision, recall, F1
- **Random seed:** `SEED = 18038391` used in all train-test splits for reproducibility

---

## Files

| File | Description |
|---|---|
| `capstone.py` | Main analysis script — all 10 questions + extra credit |
| `rmpCapstoneNum.csv` | Numerical features: ratings, difficulty, gender, pepper, etc. |
| `rmpCapstoneQual.csv` | Qualitative features: major, university, US state |
| `Capstone_Project_Report.pdf` | Full written report with figures and interpretations |

---

## How to Run

**Install dependencies:**
```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn
```

**Update file paths** in `capstone.py` (lines 20–21) to point to your local CSV files:
```python
df = pd.read_csv("rmpCapstoneNum.csv", header=None, names=cols)
qual = pd.read_csv("rmpCapstoneQual.csv", header=None, names=['major','university','state'])
```

**Run:**
```bash
python capstone.py
```

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `matplotlib` · `scipy`

---

*NYU Data Science & Economics · Spring 2025*
