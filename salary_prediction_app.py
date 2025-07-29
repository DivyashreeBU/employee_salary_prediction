import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    st.subheader("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.replace(' ?', np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None

# Data analysis function
def explore_data(df):
    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    st.subheader("📈 Class Distribution")
    st.bar_chart(df['income'].value_counts())

    st.subheader("📌 Visual Insights")

    # First Row: Gender, Education, Hours/Week
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Income by Gender")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='gender', hue='income', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("#### 🎓 Education vs Income")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x='education', hue='income', order=df['education'].value_counts().index, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    with col3:
        st.markdown("#### ⏱️ Hours/Week vs Income")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='income', y='hours-per-week', ax=ax3)
        st.pyplot(fig3)

    # Second Row: Occupation, Capital Gain
    col4, col5 ,col6= st.columns(3)

    with col4:
        st.markdown("#### 💼 Income by Occupation")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        sns.countplot(data=df, y='occupation', hue='income', order=df['occupation'].value_counts().index, ax=ax4)
        st.pyplot(fig4)

    with col5:
        st.markdown("#### 💸 Capital Gain Distribution")
        fig5, ax5 = plt.subplots()
        sns.histplot(df['capital-gain'], bins=30, kde=True, ax=ax5)
        st.pyplot(fig5)

    # Correlation at the end
    with col6:
        st.subheader("🔍 Correlation Heatmap (Numerical)")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr = df[num_cols].corr()
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

    st.subheader("💡 Feature Summary")
    st.write(df.describe(include='all'))

# Train models and compare
@st.cache_resource
def train_models(df):
    df = df.copy()
    
    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = (acc, pipe)

        if acc > best_score:
            best_score = acc
            best_model = pipe

    return results, best_model, preprocessor, num_features, cat_features

# Predict user input
def predict_input(model, preprocessor, input_df):
    probs = model.predict_proba(input_df)[0]
    pred = model.predict(input_df)[0]
    return pred, np.max(probs) * 100

# Streamlit UI
def main():
    st.set_page_config("Employee Salary Predictor", layout="wide")
    st.title("Employee Salary Prediction App")

    uploaded_file = st.file_uploader("📁 Upload a CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.replace(' ?', np.nan, inplace=True)
        df.dropna(inplace=True)
    else:
        st.warning("Please upload a dataset to proceed.")
        st.stop()

    # Train and find best model once
    results, best_model, preprocessor, num_features, cat_features = train_models(df)
    best_model_name = max(results.items(), key=lambda x: x[1][0])[0]

    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🧠 Model Comparison", "🔮 Prediction"])

    with tab1:
        explore_data(df)

    with tab2:
        st.subheader("🔬 Model Accuracy Comparison")
        for name, (acc, _) in results.items():
            st.write(f"**{name}**: {acc*100:.2f}%")
        st.success(f"✅ Best Model: {best_model_name} with {max(r[0] for r in results.values())*100:.2f}% accuracy")

    with tab3:
        st.subheader("📝 Enter Employee Details")
        st.markdown(f"🧠 **Model Used for Prediction:** {best_model_name}")

        age = st.number_input("Age", 17, 90, 30)
        workclass = st.selectbox("Workclass", df["workclass"].unique())
        education = st.selectbox("Education", df["education"].unique())
        marital_status = st.selectbox("Marital Status", df["marital-status"].unique())
        occupation = st.selectbox("Occupation", df["occupation"].unique())
        gender = st.selectbox("Gender", df["gender"].unique())
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)

        user_input = {
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [200000],
            'education': [education],
            'educational-num': [10],
            'marital-status': [marital_status],
            'occupation': [occupation],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'gender': [gender],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours_per_week],
            'native-country': ['United-States']
        }

        if st.button("Predict Salary"):
            input_df = pd.DataFrame(user_input)
            pred, confidence = predict_input(best_model, preprocessor, input_df)

            if pred == 1:
                st.success(f"💰 Predicted Salary: >50K\n\nConfidence: {confidence:.2f}%")
            else:
                st.info(f"💵 Predicted Salary: <=50K\n\nConfidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
