#!/usr/bin/env python
# coding: utf-8

# ### Humana Case Comp
# ##### Demographic Data

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# In[4]:


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


X = df.drop(['transportation_issues','person_id_syn'],axis=1)
y = df['transportation_issues']

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


# ### Random Forest

# In[8]:





# In[4]:


df = demographic_data.drop('person_id_syn',axis=1).dropna()
df=df.apply(LabelEncoder().fit_transform)


# In[5]:


X=df.drop(['transportation_issues'],axis=1)
y=df["transportation_issues"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[41]:


X_train


# In[6]:


feat_labels = list(demographic_data.columns)
feat_labels.remove("person_id_syn")
feat_labels.remove("transportation_issues")


# In[67]:


feat_labels


# In[7]:


clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)


# In[215]:


for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)


# In[8]:


sfm = SelectFromModel(clf, threshold=0.10)

# Train the selector
sfm.fit(X_train, y_train)


# In[9]:


for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])


# In[43]:


y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature Model
accuracy_score(y_test, y_pred) 


# ### RF with all demographic features
# 
#  * Using all demographic features reported an accuracy of 87%

# ### RF with reduced model

# In[1]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# In[5]:


df = demographic_data.drop('person_id_syn',axis=1).dropna()
df=df.apply(LabelEncoder().fit_transform)


# In[22]:


df


# In[6]:


X=df.drop(['transportation_issues','lang_spoken_cd','cons_hcaccprf_h'],axis=1)
y=df["transportation_issues"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[7]:


clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
classifier=clf.fit(X_train, y_train)


# In[8]:


y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature Model
accuracy_score(y_test, y_pred) 


# In[10]:


predictions = classifier.predict_proba(X_test)

print(predictions)


# ### Insights
# 
# -including only 3 features from top 10% resulted in accuracy of approx 85%
# -including all features except lang_spoken_cd resulted in accuracy of approx 86.881008%

# ### CMS Feature Selection (Radnom Forest)
# 

# In[42]:


cms_data = pd.read_csv("cms_features.csv")

cat_list=list(cms_data.select_dtypes(exclude="float64").columns)


# In[54]:


cms_data[cat_list]=cms_data[cat_list].astype("object")


# In[55]:


cms_data


# In[43]:


cms_data.info()


# In[44]:


df = cms_data.dropna()
df=df.apply(LabelEncoder().fit_transform)


# In[45]:


#X=df.drop(['transportation_issues'],axis=1)
y=df["transportation_issues"]
X=df[["cms_hospice_ind",'cms_ma_risk_score_nbr',"cms_partd_ra_factor_amt","cms_tot_partd_payment_amt","cms_rx_risk_score_nbr"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[46]:


X_train


# In[60]:


feat_labels = list(cms_data.columns)
feat_labels.remove("transportation_issues")


# In[47]:


clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)


# In[62]:


for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)


# In[48]:


y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature (4 Features) Model
accuracy_score(y_test, y_pred)


# ### Accuracy report
# 
# -full model 86.4%
# -reduced model using only top 5 most important features: 85.8%

# ### Lab Claims Features

# In[11]:


lab_data = pd.read_csv("labs_data.csv")


# In[12]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2

df=lab_data.apply(LabelEncoder().fit_transform)


# In[5]:


X = df.drop(['transportation_issues'],axis=1)
y = df['transportation_issues']


# In[6]:


chi_scores = chi2(X,y)

p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)

p_values.plot.bar() 


# #### Chi-Square Test 
# 
# Based on the chi square test, cholesterol *(lab_cholesterol_abn_result_ind)* and egfr *(lab_egfr_abn_result_ind)* do not have a relatioship with transportation issues.

# ### Random Forest Model for Lab Claims

# In[35]:


X = df[['lab_bnp_abn_result_ind']]
y=df["transportation_issues"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[24]:


X_test.head(1)


# In[36]:


clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)


# In[37]:


y_pred = clf.predict(X_test)

#Accuracy regardless of features used. Which is odd.
accuracy_score(y_test, y_pred)


# ### Accuracy Results
# 
# - full model 85.1%
# - dropped from chi sq 85.1%
# - dropped lowest 2 from clf_importance 85.1
# - top 3 from clf 85.1
# - top 2 from clf

# In[15]:


feat_labels = list(lab_data.columns)
feat_labels.remove("transportation_issues")


# In[16]:


for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)


# In[17]:


sfm = SelectFromModel(clf, threshold=0.10)

# Train the selector
sfm.fit(X_train, y_train)


# In[18]:


for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])

