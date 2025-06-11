import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🎓 Student Graduation Prediction")

# --- Baca data dari file lokal ---
@st.cache_data
def load_data():
    train_df = pd.read_excel("/main/Kelulusan Train.xlsx", engine="openpyxl")
    test_df = pd.read_excel("/main/Kelulusan Test.xlsx", engine="openpyxl")
    return train_df, test_df

train_df, test_df = load_data()
st.subheader("Sample Data (Train)")
st.dataframe(train_df.head())

# --- Preprocessing ---
def preprocess(train, test):
    train = train.copy()
    test = test.copy()

    # Pisahkan label
    y_train = train['Graduated']
    X_train = train.drop('Graduated', axis=1)

    y_test = test['Graduated']
    X_test = test.drop('Graduated', axis=1)

    # Encode kolom kategorikal
    encoders = {}
    for col in X_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    # Scaling
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = preprocess(train_df, test_df)

# --- Train & Predict ---
if st.button("Train Models and Predict"):
    with st.spinner("Training Random Forest and XGBoost..."):

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

    st.success("✅ Model Trained Successfully")

    # --- Evaluation ---
    st.subheader("🎯 Evaluation Results")

    st.markdown("**Random Forest Report**")
    st.text(classification_report(y_test, rf_pred))

    st.markdown("**XGBoost Report**")
    st.text(classification_report(y_test, xgb_pred))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, xgb_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("XGBoost Confusion Matrix")
    st.pyplot(fig)

    # --- Feature Importance ---
    st.subheader("🔍 XGBoost Feature Importance")
    importance = xgb.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(importance_df)

    fig2, ax2 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax2)
    st.pyplot(fig2)
