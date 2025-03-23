import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

# Load dataset
file_path = "Training.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["Unnamed: 133"], errors="ignore")

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

# Streamlit App Title
st.title("🩺 Disease Prediction System with AI Chatbot")
st.write("Select the symptoms you are experiencing to predict the disease.")

# Multi-select dropdown for symptoms
selected_symptoms = st.multiselect("Select Symptoms", symptom_columns)

# Prepare input data (initialize with zeros)
input_data = np.zeros(len(symptom_columns))

# Encode selected symptoms
for symptom in selected_symptoms:
    if symptom in symptom_columns:
        input_data[symptom_columns.index(symptom)] = 1

# Predict when button is clicked
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

        # Display final predicted disease
        st.success(f"🦠 **Predicted Disease:** {final_disease}")

        # Show probability distribution of top 3 diseases
        fig = px.bar(
            x=top_diseases,
            y=top_probs,
            labels={"x": "Disease", "y": "Probability"},
            title="Top 3 Most Probable Diseases",
            color=top_diseases
        )
        st.plotly_chart(fig)

# AI Chatbot Integration
OPENROUTER_API_KEY = "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"
YOUR_SITE_URL = "your-site-url"
YOUR_SITE_NAME = "your-site-name"

st.sidebar.title("🤖 AI Medical Assistant - DeepSeek Chatbot")
st.sidebar.write("Ask me about any symptom or disease!")

user_query = st.sidebar.text_input("💬 Enter your question:")

if st.sidebar.button("Chat 💬"):
    if user_query:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": YOUR_SITE_URL,
                    "X-Title": YOUR_SITE_NAME,
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [{"role": "user", "content": user_query}],
                }),
            )

            # Parse response
            if response.status_code == 200:
                result = response.json()
                st.sidebar.info(result["choices"][0]["message"]["content"])
            else:
                st.sidebar.error(f"Error: {response.text}")

        except Exception as e:
            st.sidebar.error(f"Error fetching response: {e}")
    else:
        st.sidebar.warning("⚠️ Please enter a question.")
