# Customer_segmentation
# Customer Segmentation using Hierarchical Clustering
**Live Demo:** https://customers-segment.streamlit.app/

This project implements a **customer segmentation system** using **Hierarchical Agglomerative Clustering (Ward linkage)** and deploys it as an **interactive Streamlit web application**.

The application allows users to input customer details and instantly receive a **customer segment prediction** along with a **business-friendly segment name**.



## Problem Statement
Customer segmentation helps businesses:
- Identify high-value customers
- Personalize marketing campaigns
- Improve customer retention
- Understand purchasing behavior

This project segments customers based on **demographics, spending behavior, and engagement metrics**.

---

## Model Overview

### Algorithm
- **Hierarchical Agglomerative Clustering**
  - Linkage method: `Ward`
  - Number of clusters: `3`

### Why Hierarchical Clustering?
- No need to predefine centroids
- Produces interpretable customer groups
- Ideal for exploratory customer analysis

**Note:**  
Hierarchical clustering does not support direct prediction for new data.  
To enable deployment, a **nearest-centroid assignment strategy** is used.

---

## Features Used
- Income  
- Recency  
- Customer tenure  
- Total spending  
- Total children  
- Web purchases  
- Catalog purchases  
- Store purchases  
- Web visits per month  
- Accepted campaigns  

---

## Customer Segments

| Segment ID | Segment Name |
|----------|-------------|
| 0 | High-Value Loyal Customers |
| 1 | Price-Sensitive Occasional Buyers |
| 2 | Moderate Engagement Customers |

*(Segment names are business interpretations derived from cluster characteristics.)*

---

## Project Architecture

### 🔹 Training Phase (Offline – Jupyter)
- Feature scaling using `StandardScaler`
- Hierarchical clustering with Ward linkage
- Cluster centroid computation
- Model artifacts saved as `.pkl` files

### 🔹 Deployment Phase (Streamlit)
- Load saved preprocessing and centroids
- Assign new customers to nearest centroid
- Display segment ID and segment name


