import streamlit as st
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import json
import plotly.express as px

st.set_page_config(page_title="Document Classification Dashboard", layout="wide")

# Database setup
DATABASE_URL = "sqlite:///results.db"
engine = create_engine(DATABASE_URL)

# Load data from SQLite database
@st.cache_data
def load_data():
    query = "SELECT * FROM documents"
    df = pd.read_sql(query, con=engine)
    return df

# Load data
df = load_data()

# Header
st.title("Document Classification Dashboard")
st.markdown("An interactive dashboard to analyze document classification performance")

# Convert JSON strings in `high_confidence_classes` to lists
df['high_confidence_classes'] = df['high_confidence_classes'].apply(lambda x: json.loads(x) if x else [])

# Filter out rows with missing ground truth or predictions
df = df.dropna(subset=['classified_category', 'ground_truth'])
y_pred = df['classified_category']
y_true = df['ground_truth']

# Compute summary metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
try:
    auc_score = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average="weighted", multi_class="ovr")
except ValueError:
    auc_score = None

# Organize the dashboard into four quadrants
col1, col2 = st.columns(2)

# Quadrant 1: Summary Metrics
with col1:
    st.subheader("Summary Metrics")
    st.metric("Accuracy", f"{accuracy:.2%}")
    st.metric("Weighted F1 Score", f"{f1:.2f}")
    st.metric("ROC AUC Score", f"{auc_score:.2f}" if auc_score else "N/A")

# Quadrant 2: Classification Report
with col2:
    st.subheader("Classification Report")
    classification_rep = classification_report(y_true, y_pred, output_dict=True)
    # create a table with the classification report
    st.table(classification_rep)

# Second row for distributions and misclassification analysis
col3, col4 = st.columns(2)

# Quadrant 3: Ground Truth vs Predicted Distributions
with col3:
    st.subheader("Ground Truth vs Predicted Category Distributions")
    fig1 = px.histogram(df, x="ground_truth", title="Ground Truth Distribution", color_discrete_sequence=["#636EFA"])
    fig2 = px.histogram(df, x="classified_category", title="Prediction Distribution", color_discrete_sequence=["#EF553B"])
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# Quadrant 4: Misclassifications Analysis & Error Analysis
with col4:
    st.subheader("Misclassified Instances")
    misclassified = df[df['classified_category'] != df['ground_truth']]
    if not misclassified.empty:
        st.write(f"### Misclassified samples: {len(misclassified)}")
        st.write("Examples of misclassified instances:")
        st.write(misclassified[['file_name', 'ground_truth', 'classified_category', 'high_confidence_classes']].head(5))
        
        # Error Analysis
        st.write("### Misclassification Analysis")
        error_df = misclassified.groupby(['ground_truth', 'classified_category']).size().reset_index(name='count')
        fig = px.bar(error_df, x="ground_truth", y="count", color="classified_category",
                     title="Misclassification by Category", labels={"count": "Misclassification Count"},
                     color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No misclassifications found!")
