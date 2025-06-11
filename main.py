import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="wide")
st.title("🎓 Prediksi Kelulusan Mahasiswa dengan Random Forest & XGBoost")

# Upload file
tab1, tab2 = st.tabs(["Upload Data", "Hasil Evaluasi"])

with tab1:
    train_file = st.file_uploader("Upload file Train (.xls)", type=["xls", "xlsx"], key="train")
    test_file = st.file_uploader("Upload file Test (.xls)", type=["xls", "xlsx"], key="test")

    if train_file and test_file:
        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)

        st.subheader("📄 Contoh Data Train")
        st.dataframe(train_df.head())

        # Preprocessing
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        le = LabelEncoder()
        train_df['STATUS KELULUSAN'] = le.fit_transform(train_df['STATUS KELULUSAN'])
        test_df['STATUS KELULUSAN'] = le.transform(test_df['STATUS KELULUSAN'])

        for col in ['JENIS KELAMIN', 'STATUS NIKAH', 'STATUS MAHASISWA']:
            train_df[col] = le.fit_transform(train_df[col])
            test_df[col] = le.transform(test_df[col])

        non_features = ['STATUS KELULUSAN', 'STATUS NIKAH', 'STATUS MAHASISWA', 'JENIS KELAMIN', 'NAMA']
        numerical_features = [col for col in train_df.columns if col not in non_features and train_df[col].dtype != 'object']

        scaler = MinMaxScaler()
        train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
        test_df[numerical_features] = scaler.transform(test_df[numerical_features])

        X_train = train_df.drop(columns=non_features, errors='ignore')
        y_train = train_df['STATUS KELULUSAN']
        X_test = test_df.drop(columns=non_features, errors='ignore')
        y_test = test_df['STATUS KELULUSAN']

        if st.button("🔁 Jalankan Model"):
            # Train models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)

            xgb = XGBClassifier(eval_metric='logloss', random_state=42)
            xgb.fit(X_train, y_train)
            xgb_preds = xgb.predict(X_test)

            # Evaluasi model
            def evaluate_model(y_true, y_pred, model_name):
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                spec = tn / (tn + fp)
                f1 = f1_score(y_true, y_pred)

                st.subheader(f"📊 Evaluasi Model: {model_name}")
                st.write(f"- Accuracy: {acc:.4f}")
                st.write(f"- Precision: {prec:.4f}")
                st.write(f"- Recall: {rec:.4f}")
                st.write(f"- Specificity: {spec:.4f}")
                st.write(f"- F1 Score: {f1:.4f}")

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"Confusion Matrix: {model_name}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with tab2:
                evaluate_model(y_test, rf_preds, "Random Forest")
                evaluate_model(y_test, xgb_preds, "XGBoost")

                st.subheader("📌 Feature Importance")
                importances_rf = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                importances_xgb = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)

                fig1, ax1 = plt.subplots()
                importances_rf.plot(kind='bar', ax=ax1)
                ax1.set_title("Feature Importance - Random Forest")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                importances_xgb.plot(kind='bar', ax=ax2)
                ax2.set_title("Feature Importance - XGBoost")
                st.pyplot(fig2)
    else:
        st.info("Silakan upload kedua file terlebih dahulu.")
