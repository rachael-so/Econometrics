#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np       # For numerical computations
import pandas as pd      # For data manipulation and analysis
import statsmodels.api as sm  # For econometric and statistical modeling
import scipy.stats as stats   # For statistical methods (e.g., Delta method)
import matplotlib.pyplot as plt  # For plotting (if needed for visualizations)
from tqdm import tqdm


# In[2]:


# read file into dataframe
f = "nls_2008.txt"
columns = ['luwe', 'educ', 'exper', 'age', 'fed', 'med', 'kww', 'iq', 'white']
df = pd.read_csv(f, sep='\t', header=None, names=columns)
#print(df.head(10))


# In[3]:


print("######### Problem 1 #########")
summary_stats = df.describe().loc[['min', 'max', 'mean', 'std']]
print("\n",summary_stats)


# In[4]:


print("######### Problem 2 #########")

# add column for experience-squared
df['exper_squared'] = df['exper'] ** 2

# define variables and fit model
X = sm.add_constant(df[['educ', 'exper', 'exper_squared']])
Y = df['luwe']
model = sm.OLS(Y, X).fit()
#print(model.summary())

# fit robust model
robust_model = sm.OLS(Y, X).fit(cov_type='HC0')


# In[5]:


# conventional model
print("\nConventional Estimates:")
print(model.params)
print("\nConventional Standard Errors:")
print(model.bse)

# robust model
print("\nRobust Estimates:")
print(robust_model.params)
print("\nRobust Standard Errors:")
print(robust_model.bse)


# In[6]:


# variance-covariance matrices
print("Conventional Variance-Covariance Matrix")
print(model.cov_params())
print("\nRobust Variance-Covariance Matrix")
print(robust_model.cov_params())


# In[7]:


# calculate residual variance
resid_var = np.var(model.resid, ddof=1)
robust_resid_var = np.var(robust_model.resid, ddof=1)
print("Conventional Residual Variance:", resid_var)
print("Conventional Residual Variance:", robust_resid_var)


# In[8]:


print("######### Problem 3 #########")

# new dataframe with adjusted data for -1 education level
df_educminus1 = df.copy()
df_educminus1['educ'] = df_educminus1['educ'] - 1
df_educminus1['exper'] = df_educminus1['age'] - df_educminus1['educ'] - 6
df_educminus1['exper_squared'] = df_educminus1['exper'] ** 2

# predict using problem 2 model
X_educminus1 = sm.add_constant(df_educminus1[['educ', 'exper', 'exper_squared']])
predicted_luwe_educminus1 = model.predict(X_educminus1)
#print(predicted_luwe_educminus1)

# calculate effect
avg_luwe = df['luwe'].mean()
avg_luwe_educminus1 = predicted_luwe_educminus1.mean()
change_in_log_wages = avg_luwe_educminus1 - avg_luwe
print("Effect on average log weekly wages of decreasing education by one year:", change_in_log_wages)


# In[9]:


print("######### Problem 4 #########")

# new covariates
df['new_educ'] = df['educ'] - 1
df['new_exper'] = df['exper'] + 1
df['new_exper_squared'] = df['new_exper'] ** 2

X_redefined = sm.add_constant(df[['new_educ', 'new_exper', 'new_exper_squared']])
model_redefined = sm.OLS(df['luwe'], X_redefined).fit()
#print(model_redefined.summary())
print("Effect of decreasing education by one year (new_educ coefficient):", model_redefined.params['new_educ'])


# In[10]:


print("The effect can be found but the result will be different.")


# In[11]:


print("######### Problem 5 #########")

# dataframe to reflect policy change
df_policy = df.copy()
df_policy['educ'] = df_policy['educ'].apply(lambda x: 12 if x < 12 else x)
df_policy['exper'] = df_policy['age'] - df_policy['educ'] - 6
df_policy['exper_squared'] = df_policy['exper'] ** 2

# predict using problem 2 model
X_policy = sm.add_constant(df_policy[['educ', 'exper', 'exper_squared']])
predicted_luwe_policy = model.predict(X_policy)

# calculate effect
original_avg_luwe = df['luwe'].mean()
policy_avg_luwe = predicted_luwe_policy.mean()
policy_effect = policy_avg_luwe - original_avg_luwe
print("Effect on average log weekly wages after increasing education to 12 years:", policy_effect)


# In[12]:


print("######### Problem 6 #########")

# delta method
gradient = X_policy.mean().values  # mean of the covariates represents the average change
var_conventional = np.dot(np.dot(gradient.T, model.cov_params()), gradient)
var_robust = np.dot(np.dot(gradient.T, robust_model.cov_params()), gradient)

# calculate the standard errors
se_conventional = np.sqrt(var_conventional)
se_robust = np.sqrt(var_robust)
print("Standard error (conventional):", se_conventional)
print("Standard error (robust):", se_robust)


# In[13]:


print("######### Problem 7 #########")

def bootstrap_effect(data, data_policy, model_conventional, model_robust):
    # resample data with replacement
    indices = np.random.choice(data.index, size=len(data), replace=True)
    sample = data.loc[indices]
    sample_policy = data_policy.loc[indices]
    
    # predict w/ conventional model
    predicted_luwe_policy_conventional = model.predict(sm.add_constant(sample_policy[['educ', 'exper', 'exper_squared']])).mean()
    original_luwe_mean_conventional = sample['luwe'].mean()
    effect_conventional = predicted_luwe_policy_conventional - original_luwe_mean_conventional
    
    # predict w/ robust model
    predicted_luwe_policy_robust = model_robust.predict(sm.add_constant(sample_policy[['educ', 'exper', 'exper_squared']])).mean()
    original_luwe_mean_robust = sample['luwe'].mean()
    effect_robust = predicted_luwe_policy_robust - original_luwe_mean_robust
    
    return effect_conventional, effect_robust

# run bootstraps
n_bootstraps = 100000
bootstrap_results = [bootstrap_effect(df, df_policy, model, robust_model) for _ in tqdm(range(n_bootstraps))]
bootstrap_conventional = [result[0] for result in bootstrap_results]
bootstrap_robust = [result[1] for result in bootstrap_results]

# standard error is calculated as the standard deviation of the bootstrap estimates
bootstrap_se_conventional = np.std(bootstrap_conventional)
bootstrap_se_robust = np.std(bootstrap_robust)
print("Bootstrap standard error (conventional):", bootstrap_se_conventional)
print("Bootstrap standard error (robust):", bootstrap_se_robust)


# In[14]:


print("The conventional and robust bootstrap standard errors are very similar to the analytic standard errors.")


# In[ ]:




