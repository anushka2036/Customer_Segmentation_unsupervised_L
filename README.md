# Customer Segmentation using K-Means Clustering

## 📌 Project Overview

This project uses _K-Means Clustering_ to segment customers based on their demographic and purchasing behavior.

The model analyzes features such as age, income, total spending, web purchases, store purchases, web visits, and recency to group customers into meaningful segments.

After evaluating different values of K, _K = 2_ was selected based on the highest Silhouette Score of approximately **0.324**.

### Customer Segments

* **Cluster 0 – High Value Customers:** Higher income, spending, and purchase activity.
* **Cluster 1 – Low Value Customers:** Lower income, spending, and purchase activity.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib
* Jupyter Notebook

## 📊 Model Evaluation

Different cluster values from **K=2 to K=10** were evaluated using:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

**Best K:** 2
**Silhouette Score:** ~0.324

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd customer-segmentation
```

### 2. Install required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 3. Run the Streamlit application

```bash
streamlit run app_cs.py
```

### 4. Enter customer details

Enter the customer's:

* Age
* Income
* Total Spending
* Web Purchases
* Store Purchases
* Web Visits per Month
* Recency

Click **Predict Customer Segment** to see the predicted customer group.

## 📁 Project Files

```text
customer-segmentation/
│
├── app_cs.py
├── customer_segmentation.csv
├── Customer_Segmentation.ipynb
├── kmeans_model.pkl
├── scaler.pkl
├── imputer.pkl
├── features.pkl
└── README.md
```

## 🎯 Objective

The main objective is to identify different customer groups and provide insights that can help businesses develop targeted marketing strategies, personalized offers, and customer retention plans.
