#!/usr/bin/env python
# coding: utf-8

# In[157]:


import pandas as pd
import numpy as np 
import requests
import zipfile
import io
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[158]:


data = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv")
data.head()


# In[159]:


df = data.copy()


# In[160]:


df.isnull().sum()


# In[161]:


df.dtypes


# In[162]:


categorical = list(df.dtypes[df.dtypes == "object"].index)
numerical = list(df.dtypes[df.dtypes != "object"].index)


# In[163]:


df[categorical].isnull().sum()


# In[164]:


df[numerical].isnull().sum()


# In[165]:


df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0)


# In[166]:


df[categorical].isnull().sum()


# In[167]:


df[numerical].isnull().sum()


# ## Question 1

# In[168]:


df.industry.mode()


# In[169]:


df.industry.value_counts()


# ## Question 2

# In[170]:


numerical_no_target = ['number_of_courses_viewed',
 'annual_income',
 'interaction_count',
 'lead_score']


# In[171]:


numerical


# In[172]:


# Ensure plots show inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
correlation_matrix = df[numerical_no_target].corr()

# Create the heatmap
plt.figure(figsize=(8, 6))  # Optional: make the plot larger
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Add a title
plt.title("Correlation Heatmap of Numerical Features")

# Force display
plt.show()


# >### 'annual_income' and 'interaction_count'

# In[173]:


print("interaction_count and lead_score:\t", correlation_matrix["interaction_count"]["lead_score"].item())
print("number_of_courses_viewed and lead_score:\t", correlation_matrix["number_of_courses_viewed"]["lead_score"].item())
print("number_of_courses_viewed and interaction_count:\t", correlation_matrix["number_of_courses_viewed"]["interaction_count"].item())
print("annual_income and interaction_count:\t", correlation_matrix["annual_income"]["interaction_count"].item())


# ### Splitting of the data

# In[174]:


from sklearn.model_selection import train_test_split


# In[175]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.converted.values
y_val = df_val.converted.values
y_test = df_test.converted.values

del df_train['converted']
del df_val['converted']
del df_test['converted']


# In[176]:


len(df_train), len(df_val), len(df_test)


# ## Question 3

# In[177]:


from sklearn.metrics import mutual_info_score
scores = {}

for col in categorical:
    result = mutual_info_score(y_train, df_train[col])
    scores[col] = result
    print(col, round(result, 2))

sorted(scores.items(), key=lambda x: x[1], reverse=True)


# In[178]:


scores ={}
for col in categorical:
    result = mutual_info_score(y_full_train, df_full_train[col])
    scores[col] = result
    print(col, result)
sorted(scores.items(), key=lambda x: x[1], reverse=True)


# >### 'lead_source' has the biggest mutual information score

# In[179]:


def mutual_info_y_score(series):
    return round(mutual_info_score(series, y_train), 2)

mi = df_train[categorical].apply(mutual_info_y_score)
mi.sort_values(ascending=False).index[0]


# ## Question 4

# In[180]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[181]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical_no_target].to_dict(orient='records')
val_dict = df_val[categorical + numerical_no_target].to_dict(orient='records')
test_dict = df_test[categorical + numerical_no_target].to_dict(orient='records')

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)


# In[182]:


model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_val)[:, 1]

round(((y_pred_proba >= 0.5) == y_val).mean().item(), 2)


# In[187]:


((y_pred_proba >= 0.5) == y_val).mean().item()


# In[184]:


y_pred = model.predict(X_val)

round((y_pred == y_val).mean(), 2)


# In[185]:


(y_pred == y_val).mean()


# In[186]:


training_result = model.predict(X_train)

round((training_result == y_train).mean(), 2)


# In[ ]:




