#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:21:26 2025

@author: wangtianyi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr
from statsmodels.formula.api import ols


SEED = 18038391

#rmp = pd.read_csv('/Users/wangtianyi/Downloads/college/PODS/capstone project/rmpCapstoneNum.csv')

#rmp.columns = ['Average Rating', 'Average Difficulty', 'Number of ratings', 'Received a pepper', 
          #     'proportion', ' online classes', 'Male gender', 'Female']

#make sure the column name is correct
cols = [ "avg_rating", "avg_difficulty", "num_ratings",
    "pepper", "retake_pct", "online_ratings",
    "male", "female"]

df = pd.read_csv("rmpCapstoneNum.csv", header=None, names=cols)

#clean data
print(df['num_ratings'].describe(percentiles=[.25,.5,.75,.9,.95]))
clean_df = df[df['num_ratings']>5]


# ----------------------------------- Q1 ------------------------------------------
# ----- whether there is evidence of a pro-male gender bias in this dataset -------
print("q1")
#clean data set
d1=clean_df
print(d1['male'].value_counts()) #male: 8814
print(d1['female'].value_counts()) #female: 7354
sub = clean_df[
        ((clean_df['male'] == 1)  & (clean_df['female'] == 0)) |
        ((clean_df['male'] == 0)  & (clean_df['female'] == 1))
      ].copy()
sub['gender'] = np.where(sub['male'] == 1, 'male', 'female')
print(sub['gender'].value_counts()) #male:8172 female:6721

#t-test
male   = sub.loc[sub.gender == 'male',   'avg_rating']
female = sub.loc[sub.gender == 'female', 'avg_rating']

t_stat, p_val = ttest_ind(male, female, equal_var=False)

print(f"Welch t = {t_stat:.3f},  p = {p_val:.3g}")

#control for experience
sub['log_n'] = np.log10(sub['num_ratings'])
mod = smf.ols('avg_rating ~ C(gender) + log_n', data=sub).fit()
beta  = mod.params['C(gender)[T.male]']
ci_lo, ci_hi = mod.conf_int().loc['C(gender)[T.male]']
p_reg = mod.pvalues['C(gender)[T.male]']

print("\nExperience-adjusted result:")
print(f"β_male = {beta:.3f}  (95% CI {ci_lo:.3f} … {ci_hi:.3f}),  p = {p_reg:.3e}")

#plot
sub.boxplot(column='avg_rating', by='gender',
            grid=False, showfliers=False, figsize=(4,4))
plt.suptitle('')                     # remove default super-title
plt.title('Average Rating by Gender (≥6 ratings)')
plt.ylabel('Rating (1–5)')
plt.xlabel('')
plt.tight_layout()
plt.savefig('fig_gender_rating.png', dpi=300)
plt.show()

# ----------------------------------- Q2 ------------------------------------------
print('q2')
# x = experience, y = quality
x = clean_df['num_ratings']
y = clean_df['avg_rating']

rho2, p2 = spearmanr(x, y)
print(f"Spearman ρ = {rho2:.3f},  p = {p2:.2e}")

# plot
plt.figure(figsize=(4,3))
plt.scatter(x, y, s=6, alpha=0.2)
plt.xscale('log')                         
plt.xlabel('Number of ratings (log scale)')
plt.ylabel('Average rating')
plt.title('Experience vs. Rating')

log_x = np.log10(x)
m_exp = ols('y ~ log_x', data=pd.DataFrame({'log_x': log_x, 'y': y})).fit()
xgrid = np.linspace(log_x.min(), log_x.max(), 100)
ygrid = m_exp.params['Intercept'] + m_exp.params['log_x'] * xgrid
plt.plot(10**xgrid, ygrid, color='firebrick', linewidth=2)  # transform xgrid back to log scale

plt.tight_layout()
plt.show()


# ----------------------------------- Q3 ------------------------------------------
print('q3')
x3= clean_df['avg_difficulty']       
y3 = clean_df['avg_rating']           

#correlation
r_P3, p_P3 = pearsonr(x3, y3)           
r_S3, p_S3 = spearmanr(x3, y3) 

#ols
m3 = ols('y3 ~ x3', data=clean_df).fit()
slope  = m3.params['x3']
r2_3, rmse_3 = m3.rsquared, np.sqrt(m3.mse_total)
print(f"slope = {slope:.3f},  R² = {r2_3:.3f},  RMSE = {rmse_3:.3f}")


#plot
plt.figure(figsize=(4,3))
plt.scatter(x3, y3, s=6, alpha=.25)

m3_line = ols('y3 ~ x3', data=clean_df).fit()
xgrid = np.linspace(1,5,100)
ygrid = m3_line.params['Intercept'] + m3_line.params['x3'] * xgrid
plt.plot(xgrid, ygrid, color='firebrick', linewidth=2)

plt.xlabel('Average difficulty (1 = easy, 5 = hard)')
plt.ylabel('Average rating')
plt.title('Difficulty vs. Rating')
plt.tight_layout()
plt.show()


# ----------------------------------- Q4 ----------------------------------------
print('q4')
df4 = clean_df.copy()
df4['online_ratio'] = df4['online_ratings'] / df4['num_ratings']

df4['group'] = np.where((df4['online_ratio'] >= 0.50),'online-heavy', 'mostly in-person')
print(df4['group'].value_counts())


onl = df4.loc[df4.group=='online-heavy', 'avg_rating']
f2f = df4.loc[df4.group=='mostly in-person', 'avg_rating']

# Welch t-test
t4, p4 = ttest_ind(onl, f2f, equal_var=False)


#plot
df4.boxplot(column='avg_rating', by='group',      
            grid=False, showfliers=False, figsize=(4,4))
plt.suptitle('')
plt.title('Ratings: Online-heavy vs Mostly In-person')
plt.ylabel('Average rating (1–5)')
plt.xlabel('')
plt.tight_layout()
plt.show()

# ----------------------------------- Q5 ----------------------------------------
df5 = clean_df.copy()
print('q5')
#calculate pct
if df5['retake_pct'].max() > 1:
    df5['retake_pct'] = df5['retake_pct'] / 100

xy5 = df5[['avg_rating', 'retake_pct']].dropna()
x5, y5 = xy5['avg_rating'], xy5['retake_pct']

#correlation
rP5, pP5 = pearsonr(x5, y5)
rS5, pS5 = spearmanr(x5, y5)
print(f"Pearson r = {rP5:.3f}, p = {pP5:.2e}")
print(f"Spearman ρ = {rS5:.3f}, p = {pS5:.2e}")

#ols
mod5 = ols('retake_pct ~ avg_rating', data=xy5).fit()
β1_5 = mod5.params['avg_rating']
r2_5 = mod5.rsquared
print(f"slope β1 = {β1_5:.3f},  R² = {r2_5:.3f}")

#plot
plt.figure(figsize=(4,3))
plt.scatter(x5, y5, s=6, alpha=.25)
grid = np.linspace(1,5,100)
plt.plot(grid, mod5.params[0] + β1_5*grid, color='firebrick', lw=2)
plt.ylim(0,1)
plt.xlabel('Average rating')
plt.ylabel('Prop. who would take again')
plt.title('Rating vs “Would take again”')
plt.tight_layout()
plt.show()

# ----------------------------------- Q6 ----------------------------------------
print('q6')

df6 = clean_df[['avg_rating', 'pepper']].dropna()
#clean
hot  = df6.loc[df6.pepper == 1, 'avg_rating']
not_h = df6.loc[df6.pepper == 0, 'avg_rating']

#t-test
t6, p6 = ttest_ind(hot, not_h, equal_var=False)

# effect size
n1, n2 = len(hot), len(not_h)
pooled = np.sqrt(((n1-1)*hot.var(ddof=1) + (n2-1)*not_h.var(ddof=1)) / (n1+n2-2))
d6     = (hot.mean() - not_h.mean()) / pooled

print(f"hot mean = {hot.mean():.3f}   (n = {n1})")
print(f"not mean = {not_h.mean():.3f} (n = {n2})")
print(f"Δ = {hot.mean()-not_h.mean():.3f},  Welch t = {t6:.3f},  p = {p6:.3g},  d = {d6:.3f}")


#plot
df6.boxplot(column='avg_rating', by='pepper',
            grid=False, showfliers=False, figsize=(4,4))
plt.title('Ratings: “Hot” (pepper = 1) vs Not Hot (pepper = 0)')
plt.ylabel('Average rating (1–5)')
plt.tight_layout()
plt.savefig('fig_pepper_rating.png', dpi=300)
plt.show()

# ----------------------------------- Q7 ----------------------------------------
print('q7')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df7 = clean_df[['avg_difficulty', 'avg_rating']].dropna()
X7= df7[['avg_difficulty']].values         
y7 = df7['avg_rating'].values              

X_tr7, X_te7, y_tr7, y_te7 = train_test_split(X7, y7, test_size=0.2, random_state=SEED)

lm7 = LinearRegression().fit(X_tr7, y_tr7)

y_pred7 = lm7.predict(X_te7)
r2_7   = r2_score(y_te7, y_pred7)
rmse7   = mean_squared_error(y_te7, y_pred7, squared=False)

print(f"slope β₁ (difficulty) = {lm7.coef_[0]:.3f}")
print(f"intercept β₀        = {lm7.intercept_:.3f}")
print(f"R² (test)           = {r2_7:.3f}")
print(f"RMSE (test)         = {rmse7:.3f}")

# plot
plt.figure(figsize=(4,3))
plt.scatter(X_te7, y_te7, s=6, alpha=.25)
xgrid = np.linspace(1,5,100).reshape(-1,1)
plt.plot(xgrid, lm7.predict(xgrid), color='firebrick', lw=2)
plt.xlabel('Average difficulty')
plt.ylabel('Average rating')
plt.title('Difficulty-only regression')
plt.tight_layout()
plt.show()

# ----------------------------------- Q8 ----------------------------------------
print('q8')
df8 = clean_df.copy()
df8['log_num_ratings8'] = np.log10(df8['num_ratings'])
df8['online_ratio8']    = df8['online_ratings'] / df8['num_ratings']
if df8['retake_pct'].max() > 1:
    df8['retake_pct'] = df8['retake_pct'] / 100
else:
    df8['retake_pct'] = df8['retake_pct']
df8['male_dummy8'] = df8['male'].astype(int)

X8_cols = [
    'avg_difficulty',
    'log_num_ratings8',
    'online_ratio8',
    'pepper',
    'retake_pct',
    'male_dummy8'
]
df8_model = df8[X8_cols + ['avg_rating']].dropna()
X8 = df8_model[X8_cols].values
y8 = df8_model['avg_rating'].values

X_tr8, X_te8, y_tr8, y_te8 = train_test_split(
    X8, y8, test_size=0.2, random_state=SEED
)

lr8 = LinearRegression().fit(X_tr8, y_tr8)
y_pred8 = lr8.predict(X_te8)

r2_8 = r2_score(y_te8, y_pred8)
rmse_8 = mean_squared_error(y_te8, y_pred8, squared=False)
print(f"R² (test)   = {r2_8:.3f}")
print(f"RMSE (test) = {rmse_8:.3f}")

coefs8 = pd.Series(lr8.coef_, index=X8_cols).sort_values()
print("Intercept:", lr8.intercept_)
print(coefs8)

plt.figure(figsize=(4,4))
plt.scatter(y_te8, y_pred8, s=6, alpha=0.25)
lims = [min(y_te8.min(), y_pred8.min()), max(y_te8.max(), y_pred8.max())]
plt.plot(lims, lims, '--', color='gray')
plt.xlabel('True rating')
plt.ylabel('Predicted rating')
plt.title('Predicted vs True Ratings')
plt.tight_layout()
plt.show()


# ----------------------------------- Q9 ----------------------------------------
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix)
print("q9")


df9 = clean_df[['avg_rating','pepper']].dropna()
X9   = df9[['avg_rating']].values
y9  = df9['pepper'].values

X_tr9, X_te9, y_tr9, y_te9 = train_test_split(X9, y9, test_size=0.2, stratify=y9, random_state=SEED)

clf9 = LogisticRegression(class_weight='balanced').fit(X_tr9, y_tr9)

y_prob9 = clf9.predict_proba(X_te9)[:, 1]
y_hat9  = (y_prob9 >= 0.5).astype(int)

auc9  = roc_auc_score(y_te9, y_prob9)
acc9  = accuracy_score (y_te9, y_hat9)
prec9 = precision_score(y_te9, y_hat9)
rec9  = recall_score   (y_te9, y_hat9)
f19   = f1_score       (y_te9, y_hat9)
cm9   = confusion_matrix(y_te9, y_hat9)

print(f"AUROC      = {auc9:.3f}")
print(f"accuracy   = {acc9:.3f}")
print(f"precision  = {prec9:.3f}")
print(f"recall     = {rec9:.3f}")
print(f"F1 score   = {f19:.3f}")
print("confusion matrix:\n", cm9)

fpr9, tpr9, _ = roc_curve(y_te9, y_prob9)

plt.figure(figsize=(4,4))
plt.plot(fpr9, tpr9, label=f'AUC = {auc9:.2f}')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlim(0,1); plt.ylim(0,1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ----------------------------------- Q10 ----------------------------------------
print('Q10')
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df10 = clean_df.copy()
df10['log_num_ratings'] = np.log10(df10['num_ratings'])
df10['online_ratio']    = df10['online_ratings'] / df10['num_ratings']
if df10['retake_pct'].max() > 1:
    df10['retake_pct'] /= 100
df10['male_dummy'] = df10['male'].astype(int)

X_cols10 = [
    'avg_rating','avg_difficulty','log_num_ratings',
    'online_ratio','retake_pct','male_dummy'
]
df10 = df10[X_cols10 + ['pepper']].dropna()
X10, y10 = df10[X_cols10].values, df10['pepper'].values

# 2. Train/test split (stratify to preserve class balance)
Xtr10, Xte10, ytr10, yte10 = train_test_split(
    X10, y10, test_size=0.2, stratify=y10, random_state=SEED
)

# 3. Pipeline: scale + logistic regression with balanced classes
clf10 = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED)
).fit(Xtr10, ytr10)

# 4. Predictions & metrics
yprob10 = clf10.predict_proba(Xte10)[:,1]
yhat10  = (yprob10 >= 0.5).astype(int)

auc10  = roc_auc_score(yte10, yprob10)
acc10  = accuracy_score  (yte10, yhat10)
prec10 = precision_score (yte10, yhat10)
rec10  = recall_score    (yte10, yhat10)
f110   = f1_score        (yte10, yhat10)
cm10   = confusion_matrix(yte10, yhat10)

print(f"AUROC      = {auc10:.3f}")
print(f"Accuracy   = {acc10:.3f}")
print(f"Precision  = {prec10:.3f}")
print(f"Recall     = {rec10:.3f}")
print(f"F1 score   = {f110:.3f}")
print("Confusion matrix:\n", cm10)

# 5. ROC curve
fpr10, tpr10, _ = roc_curve(yte10, yprob10)
plt.figure(figsize=(4,4))
plt.plot(fpr10, tpr10, label=f"AUC = {auc10:.2f}")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC – Pepper (all predictors, logistic)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()



# ----------------------------------- extra credit ----------------------------------------

num = df
qual = pd.read_csv('rmpCapstoneQual.csv', header=None, names=['major','university','state'])
df_ex = pd.concat([num, qual['state']], axis=1)

df_ex = df_ex[df_ex['num_ratings'] >= 6]

state_stats = (
    df_ex
    .groupby('state')
    .agg(count=('avg_rating','size'),
         mean_rating=('avg_rating','mean'))
    .query('count >= 100')
    .sort_values('mean_rating', ascending=False)
)

print("Top 5 states by rating:\n", state_stats.head(), "\n")
print("Bottom 5 states by rating:\n", state_stats.tail(), "\n")

# 3. Bar plot: top 10
plt.figure(figsize=(8,3))
state_stats['mean_rating'].head(10).plot.bar()
plt.title('Top 10 States by Mean Professor Rating (n ≥ 100)')
plt.ylabel('Mean Rating')
plt.tight_layout()
plt.show()

# 4. Bar plot: bottom 10
plt.figure(figsize=(8,3))
state_stats['mean_rating'].tail(10).plot.bar()
plt.title('Bottom 10 States by Mean Professor Rating (n ≥ 100)')
plt.ylabel('Mean Rating')
plt.tight_layout()
plt.show()
