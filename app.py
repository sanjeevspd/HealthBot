import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import boto3
import hashlib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

# AWS S3 Configuration (Hardcoded for testing, remove before production)

BUCKET_NAME = "healthbotwe3"

s3_client = boto3.client(
    "s3",
)


# Hash Password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Authentication System
def save_user(username, password):
    hashed_password = hash_password(password)
    user_data = {"username": username, "password": hashed_password}
    try:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"users/{username}.json", Body=json.dumps(user_data))
        return True
    except Exception as e:
        st.error(f"Error saving user: {e}")
        return False


def validate_user(username, password):
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=f"users/{username}.json")
        user_data = json.loads(response["Body"].read().decode("utf-8"))
        return user_data["password"] == hash_password(password)
    except Exception:
        return False


# Streamlit Authentication UI
def login_page():
    st.title("🔐 Login to Disease Prediction System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if validate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials!")



def signup_page():
    st.title("📝 Sign Up for Disease Prediction System")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    if st.button("Sign Up"):
        if save_user(new_username, new_password):
            st.success("✅ Account created successfully! Please login.")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False


# Logout function
def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["prediction"] = None  # Clear prediction state on logout
    st.session_state["chat_response"] = None  # Clear chat response state on logout
    st.rerun()


# Display Login/Signup only if user is not authenticated
if not st.session_state["authenticated"]:
    page = st.sidebar.radio("Navigation", ["Login", "Sign Up"], index=0)
    if page == "Login":
        login_page()
    elif page == "Sign Up":
        signup_page()
else:
    # Show Logout Button on Left Side
    st.sidebar.button("🚪 Logout", on_click=logout)

    # Main Application
    st.title("🩺 Disease Prediction System with AI Chatbot")
    st.write(f"Welcome, **{st.session_state['username']}**!")

    # Load dataset
    file_path = "Training.csv"
    try:
        df = pd.read_csv(file_path)
        df = df.drop(columns=["Unnamed: 133"], errors="ignore")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Separate features and target
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train classifiers
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    nb_model = GaussianNB()

    rf_model.fit(X, y_encoded)
    dt_model.fit(X, y_encoded)
    nb_model.fit(X, y_encoded)

    # Extract symptom columns
    symptom_columns = X.columns.tolist()

    # Multi-select dropdown for symptoms
    selected_symptoms = st.multiselect("Select Symptoms", symptom_columns)

    # Prepare input data
    input_data = np.zeros(len(symptom_columns))

    for symptom in selected_symptoms:
        if symptom in symptom_columns:
            input_data[symptom_columns.index(symptom)] = 1

    if st.button("🔍 Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            # Get predictions from each model
            rf_pred = rf_model.predict([input_data])[0]
            dt_pred = dt_model.predict([input_data])[0]
            nb_pred = nb_model.predict([input_data])[0]

            # Use mode to determine final prediction
            final_prediction = mode([rf_pred, dt_pred, nb_pred], keepdims=True).mode[0]

            # Decode the final prediction
            final_disease = label_encoder.inverse_transform([final_prediction])[0]

            # Get probability distributions for each model
            rf_probs = rf_model.predict_proba([input_data])[0]
            dt_probs = dt_model.predict_proba([input_data])[0]
            nb_probs = nb_model.predict_proba([input_data])[0]

            # Combine probabilities
            all_probs = (rf_probs + dt_probs + nb_probs) / 3  # Averaging probabilities

            # Get top 3 diseases with highest probabilities
            top_indices = np.argsort(all_probs)[-3:][::-1]
            top_diseases = label_encoder.inverse_transform(top_indices)
            top_probs = all_probs[top_indices]

            # Store prediction in session state
            st.session_state["prediction"] = {"disease": final_disease, "top_diseases": top_diseases,
                                              "top_probs": top_probs}

    # Display stored prediction
    if "prediction" in st.session_state and st.session_state["prediction"]:
        pred_data = st.session_state["prediction"]
        st.success(f"🦠 **Predicted Disease:** {pred_data['disease']}")

        # Show probability distribution of top 3 diseases
        fig = px.bar(
            x=pred_data["top_diseases"],
            y=pred_data["top_probs"],
            labels={"x": "Disease", "y": "Probability"},
            title="Top 3 Most Probable Diseases",
            color=pred_data["top_diseases"]
        )
        st.plotly_chart(fig)

    # Chatbot
    OPENROUTER_API_KEY = "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"
    YOUR_SITE_URL = "http://localhost:8501/"
    YOUR_SITE_NAME = "Disease Prediction AI"

    st.sidebar.title("🤖 AI Medical Assistant - DeepSeek Chatbot")
    st.sidebar.write("Ask me about any symptom or disease!")

    user_query = st.sidebar.text_input("💬 Enter your question:")

    if st.sidebar.button("Chat 💬"):
        if user_query:
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": YOUR_SITE_URL,
                    "X-Title": YOUR_SITE_NAME,
                }

                payload = {
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [{"role": "user", "content": user_query}],
                }

                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state["chat_response"] = result["choices"][0]["message"]["content"]

            except Exception as e:
                st.session_state["chat_response"] = f"Error fetching response: {e}"

    if "chat_response" in st.session_state and st.session_state["chat_response"]:
        st.sidebar.info(st.session_state["chat_response"])
