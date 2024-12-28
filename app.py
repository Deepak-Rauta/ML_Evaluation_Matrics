from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import pickle as pk
# File path to the dataset
file_path = r"C:\Users\kdeepak_new\Downloads\preprocessd_data.csv"

# Streamlit app
st.title("Machine Learning Evaluation Matrices")

# Model selection
model_name = st.selectbox(
    "select a Ml Model",
    ["Decision Tree", "Logistic Regression", "Random Forest", "Gausian NB", "KNN"]
)

# Select a Model
if model_name:
    st.header(f"Evaluations Metrics for {model_name}")

# Load the Dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X , y

X, y = load_data(file_path)

# Train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to load and save the model
def load_model(model_name):
    model_paths = {
        "Decision Tree": "Decision_tree_model.pkl",
        "Logistic Regression": "logistic_model.pkl",
        "Random Forest": "random_model.pkl",
        "Gausian NB": "naive_model.pkl",
        "KNN": "knn_model.pkl"
    }
    model_path = model_paths.get(model_name)
    if model_path:
        with open(model_path, 'rb') as file:
            model = pk.load(file)
        return model
    else:
        return None
# load the selected model
model = load_model(model_name)

# get the selected model
if model:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    # Calculate metrics
    con_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display matrix
    st.text("Confusion matrix:")
    st.table(
        pd.DataFrame(
            con_matrix,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"],
        )
    )
    st.text(f"precision: {precision}")
    st.text(f"recall: {recall}")
    st.text(f"f1_score: {f1}")
    st.text(f"Accuracy: {accuracy}")

    # Display metrics
    st.subheader("Evaluation Metrics")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

    # Plot ROC Curve
    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {model_name}")
        plt.legend(loc="lower right")
        st.pyplot(plt)
    else:
        st.warning("This model does not support probability predictions.")
else:
    st.error("Model could not be loaded. Please check the model files.")





