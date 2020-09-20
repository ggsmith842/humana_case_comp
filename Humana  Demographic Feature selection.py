#!/usr/bin/env python
# coding: utf-8

# ### Humana Case Comp
# ##### Demographic Data

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[131]:


demographic_data = pd.read_csv("demographic_data.csv")
demographic_data.head()


# In[119]:


#summary stats for state
pd.Series(demographic_data.groupby("state_cd").transportation_issues.sum()).plot(kind='bar')


# ### Feature selection for Demographic Categorical Varaibles

# In[132]:


#cons_n65p_y	cons_online_buyer	cons_ret_y	cons_retail_buyer	cons_veteran_y

df = demographic_data.drop(["est_age",'cons_n2mob','cons_n2pbl','cons_n2pmv'],axis=1).dropna()
df


# In[100]:


label_encoder = LabelEncoder()
df['cons_cmys'] = label_encoder.fit_transform(df['cons_cmys'])
df['sex_cd'] = label_encoder.fit_transform(df['sex_cd'])
df['cnty_cd'] = label_encoder.fit_transform(df['cnty_cd'])
df['state_cd'] = label_encoder.fit_transform(df['state_cd'])
df['cons_cmys'] = label_encoder.fit_transform(df['cons_cmys'])
df['cons_hhcomp'] = label_encoder.fit_transform(df['cons_hhcomp'])
df['cons_homstat'] = label_encoder.fit_transform(df['cons_homstat'])
df['lang_spoken_cd'] = label_encoder.fit_transform(df['lang_spoken_cd'])
df['cons_hcaccprf_h'] = label_encoder.fit_transform(df['cons_hcaccprf_h'])
df['cons_hcaccprf_p'] = label_encoder.fit_transform(df['cons_hcaccprf_p'])
df['cons_n2029_y'] = label_encoder.fit_transform(df['cons_n2029_y'])
df['cons_n65p_y'] = label_encoder.fit_transform(df['cons_n65p_y'])
df['cons_online_buyer'] = label_encoder.fit_transform(df['cons_online_buyer'])
df['cons_ret_y'] = label_encoder.fit_transform(df['cons_ret_y'])
df['cons_n65p_y'] = label_encoder.fit_transform(df['cons_n65p_y'])


# In[129]:


from sklearn.preprocessing import OneHotEncoder
df=OneHotEncoder().fit_transform(df)


# In[134]:


df=df.apply(LabelEncoder().fit_transform)


# In[135]:


from sklearn.feature_selection import chi2


# In[137]:


X = df.drop(['transportation_issues','person_id_syn'],axis=1)
y = df['transportation_issues']


# In[ ]:





# In[138]:


X


# In[139]:


chi_scores = chi2(X,y)

p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)

p_values.plot.bar() 


# ### Conclusion for Demographic Categorical Features
# 
#  *  all features with the exception of language spoken and cons_hcaccprf_p seem to play a role in predicting transportation using a significance level of 0.5
#  

# ### Feature Significance for Numeric Categorical

# In[160]:


categorical_cols=list(df.columns)
categorical_cols.remove("transportation_issues")
categorical_cols
demographic_numeric=demographic_data.drop(categorical_cols,axis=1).dropna()


# In[161]:


demographic_numeric.corr()


# In[166]:


ax=sns.heatmap(demographic_numeric.corr(),annot=True)
top,bottom=ax.get_ylim()
ax.set_ylim(top+0.5,bottom-0.5)


# In[162]:


X=demographic_numeric.drop("transportation_issues",axis=1)
y=demographic_numeric["transportation_issues"]


# In[163]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[168]:


demographic_data.head()

