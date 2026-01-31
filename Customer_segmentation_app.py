#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[2]:


import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin

# -----------------------------
# Load artifacts
# -----------------------------
scaler = joblib.load("scaler.pkl")
centroids = joblib.load("hierarchical_centroids.pkl")

clustering_features = [
    'income',
    'recency',
    'customer_tenure',
    'total_spending',
    'total_children',
    'numwebpurchases',
    'numcatalogpurchases',
    'numstorepurchases',
    'numwebvisitsmonth',
    'total_accepted_campaigns'
]

SEGMENT_NAMES = {
    0: "High-Value Loyal Customers",
    1: "Price-Sensitive Occasional Buyers",
    2: "Moderate Engagement Customers"
}

# -----------------------------
# Predict cluster
# -----------------------------
def predict_cluster(customer_dict):
    df = pd.DataFrame([customer_dict])
    X = df[clustering_features]
    X_scaled = scaler.transform(X)

    # assign nearest centroid
    cluster_id = pairwise_distances_argmin(X_scaled, centroids)[0]
    return int(cluster_id)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Customer Segmentation")

st.title(" Customer Segmentation (Hierarchical)")
st.write("Ward Linkage Model")

income = st.number_input("Income", min_value=0)
recency = st.number_input("Recency", min_value=0)
customer_tenure = st.number_input("Customer Tenure", min_value=0)
total_spending = st.number_input("Total Spending", min_value=0)
total_children = st.number_input("Total Children", min_value=0)
numwebpurchases = st.number_input("Web Purchases", min_value=0)
numcatalogpurchases = st.number_input("Catalog Purchases", min_value=0)
numstorepurchases = st.number_input("Store Purchases", min_value=0)
numwebvisitsmonth = st.number_input("Web Visits per Month", min_value=0)
total_accepted_campaigns = st.number_input("Accepted Campaigns", min_value=0)

if st.button("Predict Segment"):
    customer = {
        'income': income,
        'recency': recency,
        'customer_tenure': customer_tenure,
        'total_spending': total_spending,
        'total_children': total_children,
        'numwebpurchases': numwebpurchases,
        'numcatalogpurchases': numcatalogpurchases,
        'numstorepurchases': numstorepurchases,
        'numwebvisitsmonth': numwebvisitsmonth,
        'total_accepted_campaigns': total_accepted_campaigns
    }

    segment = predict_cluster(customer)

    st.success(f" Segment {segment}: {SEGMENT_NAMES[segment]}")


# In[ ]:




