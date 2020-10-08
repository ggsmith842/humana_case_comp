#!/usr/bin/env python
# coding: utf-8

# ## Clean Data
# 
#  - compiled most significant variables from each category
#      - restricted to top 5 for some categories to lower feature count to prevent overfitting
#      - trying to limit number of features to at most 50
#  - all missing values for each feature has been replaced
#      - mean for numeric
#      - mode for categorical
#      - changed data type for binary features to accound for NaNs 
#  - will standardize all numeric features
#  
#  
#  Final deliverable wil be a dataset with the most important features, no missing values, and standardized numeric variables.

# In[1]:


import pandas as pd
import numpy as np


# ### Training Data

# In[215]:


training_data = pd.read_csv('humana_train.csv')


# In[216]:


training_data.head()


# In[242]:


columns_used = ['person_id_syn', 'transportation_issues',
                'est_age','cons_homstat','cons_n2029_y',
                'cons_n2mob','cms_hospice_ind',
                'cms_ma_risk_score_nbr',
                'cms_partd_ra_factor_amt',
                'cms_tot_partd_payment_amt',
                'cms_rx_risk_score_nbr',
               'rx_branded_pmpm_ct',
                'rx_bh_pmpm_ct',
                'rx_overall_pmpm_ct',
                'rx_maint_pmpm_ct',
                'rx_generic_pmpm_ct',
               'betos_o1a_pmpm_ct',
                'credit_bal_1stmtg_30to59dpd',
                'credit_bal_1stmtg_severederog',
                'credit_bal_autobank',
                'credit_bal_autobank_new',
                'credit_bal_bankcard_severederog',
                'credit_hh_1stmtgcredit',
                'credit_hh_bankcardcredit_60dpd',
                'credit_hh_consumerfinance',
                'credit_hh_nonmtgcredit_60dpd',
                'credit_hh_totalallcredit_bankruptcy',
                'credit_hh_totalallcredit_collections',
                'credit_hh_totalallcredit_severederog',
                'credit_minmob_mtgcredit',
                'credit_num_bankcard_severederog',
                'med_ambulance_visit_ct_pmpm',
                'med_er_visit_ct_pmpm',
                'submcc_cir_hbp_pmpm_ct',
                'submcc_dia_othr_pmpm_ct',
                'submcc_men_depr_pmpm_ct',
                'submcc_men_othr_pmpm_ct',
                'submcc_mus_back_pmpm_ct',
                'submcc_ner_othr_pmpm_ct',
                'submcc_res_copd_pmpm_ct',
                'submcc_rsk_smok_pmpm_ct',
                'submcc_vco_othr_pmpm_ct',
                'dcsi_score']

len(columns_used) #gotta add condition


# In[243]:


df=training_data[columns_used]
df[['transportation_issues','cons_n2029_y','cms_hospice_ind','dcsi_score']]=df[['transportation_issues','cons_n2029_y','cms_hospice_ind','dcsi_score']].astype('object')
df.head()


# In[201]:


#replace missing values
def fill_missing(df):
    for i in df:
        if df[i].dtypes == object:
            fill = df[i].mode().iat[0]
            df.loc[:,i] = df[i].replace(np.nan,fill)
        else:
            the_mean = df[i].mean(skipna=True)
            df.loc[:,i] = df[i].replace(np.nan,the_mean)


# In[244]:


fill_missing(df)


# In[245]:


df.head()


# In[246]:


df.isna().sum()


# In[236]:


df.to_csv('cleaned_training.csv') 


# ## Holdout Data

# In[237]:


holdout = pd.read_csv('2020_Competition_Holdout .csv')


# In[238]:


holdout.head()


# In[247]:


columns_used.pop(1)


# In[248]:


df=holdout[columns_used]
df[['cons_n2029_y','cms_hospice_ind','dcsi_score']]=df[['cons_n2029_y','cms_hospice_ind','dcsi_score']].astype('object')


# In[249]:


df.head()


# In[250]:


fill_missing(df)


# In[251]:


df.head()
df.isna().sum()


# In[252]:


df.to_csv('cleaned_holdout.csv') 

