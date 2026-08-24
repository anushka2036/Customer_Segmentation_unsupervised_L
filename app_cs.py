import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Segmentation",
    layout="centered"
)

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
features = joblib.load("features.pkl")

cluster_names = {
    0: "High Value Customers",
    1: "Low Value Customers"
}

st.title("Customer Segmentation App")

st.write(
    "Enter the customer details below to identify the "
    "customer segment using the K-Means clustering model."
)

st.info(
    "The model uses K-Means clustering with 2 customer segments."
)


st.subheader("Customer Information")


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)


income = st.number_input(
    "Income",
    min_value=0.0,
    max_value=200000.0,
    value=50000.0,
    step=1000.0
)


total_spending = st.number_input(
    "Total Spending",
    min_value=0.0,
    max_value=5000.0,
    value=1000.0,
    step=50.0
)


num_web_purchases = st.number_input(
    "Number of Web Purchases",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)


num_store_purchases = st.number_input(
    "Number of Store Purchases",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)


num_web_visits = st.number_input(
    "Number of Web Visits Per Month",
    min_value=0,
    max_value=50,
    value=10,
    step=1
)


recency = st.number_input(
    "Recency (Days)",
    min_value=0,
    max_value=365,
    value=30,
    step=1
)


input_data = pd.DataFrame(
    [[
        age,
        income,
        total_spending,
        num_web_purchases,
        num_store_purchases,
        num_web_visits,
        recency
    ]],
    columns=features
)

if st.button(
    "Predict Customer Segment",
    type="primary"
):


    input_imputed = imputer.transform(
        input_data
    )

    input_scaled = scaler.transform(
        input_imputed
    )


    cluster = kmeans.predict(
        input_scaled
    )[0]


    segment_name = cluster_names.get(
        cluster,
        "Unknown Segment"
    )


    st.success(
        f"Predicted Segment: {segment_name}"
    )

    st.write(
        f"**Cluster Number:** {cluster}"
    )


    if cluster == 0:

        st.markdown(
            """
            ### 💎 High Value Customer

            This customer belongs to the high-value segment.

            **Typical characteristics:**
            - Higher income
            - Higher total spending
            - More web purchases
            - More store purchases
            - Strong purchasing behavior

            **Business strategy:**  
            Focus on customer retention, premium offers,
            loyalty programs, and personalized promotions.
            """
        )

    elif cluster == 1:

        st.markdown(
            """
            ### 🛍️ Low Value Customer

            This customer belongs to the low-value segment.

            **Typical characteristics:**
            - Lower income
            - Lower total spending
            - Fewer purchases
            - Lower purchasing activity
            - Higher website visits relative to purchases

            **Business strategy:**  
            Use targeted discounts, personalized recommendations,
            introductory offers, and conversion-focused campaigns.
            """
        )


    with st.expander("View Customer Input"):

        st.dataframe(
            input_data,
            use_container_width=True
        )