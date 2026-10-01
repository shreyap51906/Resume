import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TARGET = "Placement"
DEFAULT_FEATURES = ["IQ", "Prev_Sem_Result", "CGPA"]
DEMO = {"IQ": 120.0, "Prev_Sem_Result": 8.2, "CGPA": 8.5}  # demo student

st.set_page_config(page_title="Placement Predictor", layout="wide")
st.title("🎓 Student Placement Prediction")

# ============ 1. DATASET ============
uploaded = st.sidebar.file_uploader("Upload dataset CSV", type="csv")
if uploaded is None:
    st.info("Upload the Kaggle placement dataset CSV in the sidebar to begin.")
    st.stop()
raw = pd.read_csv(uploaded)
raw.columns = raw.columns.str.strip()


# ============ 2. DATA CLEANING ============
@st.cache_data
def clean(df):
    df = df.copy()
    log = {"Rows before": len(df)}
    log["Duplicate rows removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log["Rows with missing values removed"] = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    # Encode Yes/No columns as 1/0
    for c in df.select_dtypes("object"):
        if set(df[c].unique()) <= {"Yes", "No"}:
            df[c] = df[c].map({"No": 0, "Yes": 1})
    log["Rows after"] = len(df)
    return df, log


df, clean_log = clean(raw)
candidates = [c for c in df.select_dtypes("number").columns
              if c not in (TARGET, "College_ID")]

# ============ 3. FEATURE SELECTION (sidebar) ============
st.sidebar.header("Feature selection")
features = st.sidebar.multiselect(
    "Features used by the models", candidates,
    default=[f for f in DEFAULT_FEATURES if f in candidates])
if not features:
    st.warning("Select at least one feature.")
    st.stop()


# ============ 4-6. SPLIT, SCALE, TRAIN ============
@st.cache_resource
def train(data, feats):
    X, y = data[list(feats)], data[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = make_pipeline(StandardScaler(),  # scaler fit on train only
                         LogisticRegression(class_weight="balanced", max_iter=1000))
    pipe.fit(Xtr, ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    return {"Logistic Regression": pipe}, acc, yte, len(Xtr), len(Xte)


models, acc, yte, ntr, nte = train(df, tuple(features))
final_name = "Logistic Regression"  # ============ FINAL MODEL ============
final_model = models[final_name]

tabs = st.tabs(["1️⃣ Dataset & Cleaning", "2️⃣ EDA", "3️⃣ Features & Split",
                "4️⃣ Model & Accuracy", "5️⃣ Final Model & Prediction"])

# ---- Dataset & Cleaning ----
with tabs[0]:
    st.subheader("Dataset")
    st.dataframe(raw.head(10))
    st.write(f"Shape: {raw.shape[0]} rows × {raw.shape[1]} columns")
    st.subheader("Cleaning steps")
    st.table(pd.Series(clean_log, name="Count").to_frame())
    st.write("Missing values per column (original data):")
    st.dataframe(raw.isna().sum().rename("Missing").to_frame().T)

# ---- EDA ----
with tabs[1]:
    st.subheader("Statistics")
    st.dataframe(df[candidates + [TARGET]].describe().T.round(2))

    st.subheader("Graphs")
    st.write("Placement class balance")
    st.bar_chart(df[TARGET].map({0: "Not placed", 1: "Placed"}).value_counts())

    st.write("Feature distributions (placed vs not placed)")
    cols = st.columns(3)
    for i, f in enumerate(features):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(df[df[TARGET] == 0][f], bins=20, alpha=0.6, label="Not placed")
        ax.hist(df[df[TARGET] == 1][f], bins=20, alpha=0.6, label="Placed")
        ax.set_title(f)
        ax.legend(fontsize=7)
        cols[i % 3].pyplot(fig)

    st.write("Correlation heatmap")
    corr = df[candidates + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im)
    st.pyplot(fig)

# ---- Features & Split ----
with tabs[2]:
    st.subheader("Feature selection")
    st.write("Correlation of each feature with Placement (higher = more useful):")
    st.bar_chart(df[candidates].corrwith(df[TARGET]).abs().sort_values(ascending=False))
    st.write(f"**Selected features:** {', '.join(features)}  (change them in the sidebar)")
    st.subheader("Train/Test split")
    st.write(f"80% train ({ntr} rows) / 20% test ({nte} rows), stratified so both "
             "sets keep the same placed/not-placed ratio.")
    st.subheader("Feature scaling")
    st.write("StandardScaler is fitted on the training data only, then applied to the "
             "test data and new students (this prevents data leakage). It matters for "
             "Logistic Regression.")

# ---- Models & Evaluation ----
with tabs[3]:
    st.subheader("Model: Logistic Regression")
    st.metric("Accuracy (20% test set)", f"{acc * 100:.1f}%")
    st.progress(float(acc))
    st.write(f"The model predicted {round(acc * nte)} of {nte} test students correctly.")

# ---- Final model & prediction ----
with tabs[4]:
    st.subheader(f"Final model: {final_name}")
    st.write(f"Test accuracy: {acc * 100:.1f}%")

    st.subheader("Predict for a new student")
    inputs = {}
    cols = st.columns(len(features))
    for col, f in zip(cols, features):
        lo, hi = float(df[f].min()), float(df[f].max())
        default = DEMO.get(f, float(df[f].median()))
        inputs[f] = col.number_input(f, min_value=lo, max_value=hi,
                                     value=min(max(default, lo), hi))
    student = pd.DataFrame([inputs])
    prob = final_model.predict_proba(student)[0][1]
    label = "YES" if final_model.predict(student)[0] == 1 else "NO"

    c1, c2 = st.columns(2)
    c1.metric("Placement Prediction", label)
    c2.metric("Placement Percentage", f"{prob * 100:.1f}%")
    st.progress(float(prob))
    (st.success if label == "YES" else st.error)(f"Placement Prediction: {label}")
