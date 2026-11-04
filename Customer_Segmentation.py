
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

st.title("Customer Segmentation Dashboard")
st.write("Analyze customer demographics and purchasing behavior.")

# Generate sample customer data
def generate_data():
    np.random.seed(42)

    return pd.DataFrame({
        "CustomerID": range(1, 301),
        "Age": np.random.randint(18, 65, 300),
        "Annual_Income": np.random.randint(15000, 120001, 300),
        "Spending_Score": np.random.randint(1, 101, 300)
    })

# Upload customer data
st.sidebar.header("Upload Customer Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    df = generate_data()
    st.sidebar.info("Using sample data.")

# Check required columns
required_columns = [
    "Age",
    "Annual_Income",
    "Spending_Score"
]

if not all(column in df.columns for column in required_columns):
    st.error(
        "Required columns: Age, Annual_Income, Spending_Score"
    )
    st.stop()

# Clean data
for column in required_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

df = df.dropna(subset=required_columns).copy()

if len(df) < 2:
    st.error("At least two valid customer records are required.")
    st.stop()

# Select number of clusters
st.sidebar.header("Clustering Settings")

k = st.sidebar.slider(
    "Number of Customer Segments",
    min_value=2,
    max_value=min(10, len(df)),
    value=min(5, len(df))
)

# Perform K-Means clustering
features = df[required_columns]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Segment"] = model.fit_predict(scaled_data)

# Display KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric(
    "Average Income",
    f"${df['Annual_Income'].mean():,.0f}"
)
col3.metric(
    "Average Spending Score",
    f"{df['Spending_Score'].mean():.1f}"
)
col4.metric("Customer Segments", k)

st.divider()

# Customer segmentation chart
st.subheader("Customer Segmentation")

fig = px.scatter(
    df,
    x="Annual_Income",
    y="Spending_Score",
    color=df["Segment"].astype(str),
    hover_data=["CustomerID", "Age"],
    title="Income vs Spending Score"
)

st.plotly_chart(fig, use_container_width=True)

# Segment summary
st.subheader("Customer Segment Analysis")

summary = df.groupby("Segment").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual_Income", "mean"),
    Average_Spending=("Spending_Score", "mean")
).reset_index()

summary = summary.round(2)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

# Customer distribution chart
fig2 = px.bar(
    summary,
    x="Segment",
    y="Customer_Count",
    color="Segment",
    title="Number of Customers in Each Segment"
)

st.plotly_chart(fig2, use_container_width=True)

# Filter customers
st.subheader("Explore Customer Groups")

selected_segment = st.selectbox(
    "Choose a Segment",
    sorted(df["Segment"].unique())
)

filtered_df = df[df["Segment"] == selected_segment]

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# Download report
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Customer Segmentation Report",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)
