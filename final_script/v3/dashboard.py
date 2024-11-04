import streamlit as st
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import json
import plotly.express as px
from config.base_config import BaseConfig
import plotly.graph_objects as go

st.set_page_config(page_title="Document Classification Dashboard", layout="wide")

# Database setup
DATABASE_URL = BaseConfig.DATABASE_URL
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

# Add a new row for process metadata visualizations
st.markdown("---")
st.subheader("Process Metadata Analysis")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Processing Time Distribution")

    # Extract processing times
    processing_times = []
    for metadata_str in df['process_metadata']:
        try:
            if metadata_str and isinstance(metadata_str, str):
                metadata = json.loads(metadata_str)
                times = {
                    'OCR': metadata.get('OCR', {}).get('time', 0),
                    'Text Cleaning': metadata.get('Text Cleaning', {}).get('time', 0),
                    'Classification': metadata.get('Classification', {}).get('time', 0)
                }
                processing_times.append(times)
            elif isinstance(metadata_str, dict):  # If it's already a dictionary
                times = {
                    'OCR': metadata_str.get('OCR', {}).get('time', 0),
                    'Text Cleaning': metadata_str.get('Text Cleaning', {}).get('time', 0),
                    'Classification': metadata_str.get('Classification', {}).get('time', 0)
                }
                processing_times.append(times)
        except (json.JSONDecodeError, AttributeError) as e:
            st.warning(f"Skipping invalid metadata entry: {e}")
            continue
    
    time_df = pd.DataFrame(processing_times)
    
    # Create box plot for processing times
    fig_box = go.Figure()
    for column in time_df.columns:
        fig_box.add_trace(go.Box(y=time_df[column], name=column))
    
    fig_box.update_layout(
        title="Distribution of Processing Times by Stage",
        yaxis_title="Time (seconds)",
        showlegend=True
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Add average processing time metrics
    st.write("Average Processing Times:")
    for column in time_df.columns:
        st.metric(f"{column} Avg Time", f"{time_df[column].mean():.2f}s")

with col6:
    st.subheader("Cost Analysis")
    
    costs = []
    for metadata_str in df['process_metadata']:
        try:
            if isinstance(metadata_str, str):
                metadata = json.loads(metadata_str)
            else:
                metadata = metadata_str
            cost = metadata.get('Classification', {}).get('cost', 0)
            costs.append(cost)
        except (json.JSONDecodeError, AttributeError) as e:
            costs.append(0)
    # Create histogram for cost distribution
    fig_cost = px.histogram(
        x=costs,
        title="Distribution of Classification Costs",
        labels={'x': 'Cost ($)', 'y': 'Count'},
        color_discrete_sequence=["#00CC96"]
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # Add cost metrics
    total_cost = sum(costs)
    avg_cost = total_cost / len(costs) if costs else 0
    st.metric("Total Classification Cost", f"${total_cost:.4f}")
    st.metric("Average Cost per Document", f"${avg_cost:.4f}")
    
    # Create timeline plot of processing times
    timeline_df = pd.DataFrame(processing_times).cumsum()
    fig_timeline = px.line(
        timeline_df,
        title="Cumulative Processing Time by Stage",
        labels={'value': 'Cumulative Time (s)', 'variable': 'Stage'},
    )
    st.plotly_chart(fig_timeline, use_container_width=True)