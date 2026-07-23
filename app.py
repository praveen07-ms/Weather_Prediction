import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("seattle-weather.csv")
df = df.drop_duplicates()

# Features and target
x = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather']

# Train model
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Streamlit UI
st.title("Weather Prediction App")

st.write("Model Accuracy:", model.score(x_test, y_test))

precipitation = st.number_input("Precipitation")
temp_max = st.number_input("Maximum Temperature")
temp_min = st.number_input("Minimum Temperature")
wind = st.number_input("Wind Speed")

if st.button("Predict Weather"):

    input_data = pd.DataFrame(
        [[precipitation, temp_max, temp_min, wind]],
        columns=['precipitation', 'temp_max', 'temp_min', 'wind']
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Weather: {prediction[0]}")
    