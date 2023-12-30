
import streamlit as st
import dill
import pandas as pd

# Load the model and pipeline
def load_model():
    with open('model_and_pipeline.joblib', 'rb') as file:
        data = dill.load(file)
    return data

data = load_model()
model = data["model"]
pipeline = data["pipeline"]

# Define the Streamlit app layout
st.set_page_config(
    page_title="🔮 Customer Churn Prediction",
    page_icon="💳",
    layout="centered",
)

st.title("🔮 Customer Churn Prediction ")

# Define the input fields for each feature
credit_score = st.slider("CreditScore 🌐", min_value=300, max_value=850, step=1)
geography_options = ['France', 'Germany', 'Spain']
geography = st.selectbox("Geography 🗺️", geography_options)
gender = st.selectbox("Gender 👫", ["Male 👨", "Female 👩"])
age = st.slider("Age 🎂", min_value=18, max_value=100, step=1)
tenure = st.slider("Tenure (Number of Years) ⏳", min_value=1, max_value=20, step=1)
balance = st.number_input("Balance 💰", min_value=0.0, format="%.2f")
num_of_products = st.slider("Number of Products 📦", min_value=1, max_value=4, step=1)
has_credit_card = st.slider("Has Credit Card 💳 (Type 0 for 'No', 1 for 'Yes')", min_value=0, max_value=1)
is_active_member = st.slider("Active Member 🏃‍♂️(Type 0 for 'No', 1 for 'Yes')", min_value=0, max_value=1)
estimated_salary = st.number_input("Estimated Salary 💵", min_value=0.0, format="%.2f")
complain = st.slider("Customer has Complaint ❓(Type 0 for 'No', 1 for 'Yes')", min_value=0, max_value=1)
satisfaction_score = st.slider("Satisfaction Score 😊", min_value=1, max_value=5, step=1)
card_type_options = ["Diamond","Gold 🏅", "Platinum 🌟", "Silver 🥈"]
card_type = st.selectbox("Card Type 💳", card_type_options)
points_earned = st.number_input("Points Earned 🎉", min_value=0)

# Function to make predictions
def make_prediction():
    # Create a DataFrame from the user inputs
    user_inputs_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [str(has_credit_card)],
        'IsActiveMember': [str(is_active_member)],
        'EstimatedSalary': [estimated_salary],
        'Complain': [str(complain)],
        'Satisfaction Score': [satisfaction_score],
        'Card Type': [card_type],
        'Point Earned': [points_earned]
    })
   
    # You may need to preprocess the input data using your pipeline
    if pipeline is not None:
        processed_inputs = pipeline.transform(user_inputs_df)
    else:
        processed_inputs = user_inputs_df  # If no preprocessing is needed

    prediction = model.predict(processed_inputs)
    return prediction[0]

if st.button("🔮 Predict"):
    prediction = make_prediction()
    st.subheader("📊 Prediction Result:")
    if prediction == 0:
        st.success("✅ The customer is not expected to exit the bank.")
        st.image('not churned.jpg', caption='Prediction 0 Image', use_column_width=True)
        st.balloons()
    elif prediction == 1:
        st.error("❌ The customer is expected to exit the bank.")
        st.image('churned.jpg', caption='Prediction 1 Image', use_column_width=True)
        st.warning("💔 Please review the user inputs and consider taking appropriate action.")

   
