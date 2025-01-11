#performinf EDA functions
#applying  a ml_model
#using streamlit
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('C:/Users/LENOVO/Downloads/archive/seattle-weather.csv')

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract features from the date
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday  # 0 = Monday, 6 = Sunday

# Drop the 'date' column
data = data.drop('date', axis=1)

# Example: Encoding 'weather' column using Label Encoding
label_encoder = LabelEncoder()
data['weather_encoded'] = label_encoder.fit_transform(data['weather'])

# Drop the original 'weather' column as we now have the encoded version
data = data.drop('weather', axis=1)

# Assuming 'temp_max' is your target column
X = data.drop('temp_max', axis=1)  # Features (dropping target column)
y = data['temp_max']  # Target (temperature)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# Set the theme for the app
st.set_page_config(page_title="Weather Prediction App", page_icon="🌤", layout="wide")

# Custom CSS to style the page
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(45deg, #0892d0, #4b0082); /* Blue to purple gradient */
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            color: white;  /* Ensures text is readable on the gradient background */
            font-size: 30px;
        }

        .title {
            text-align: center;
            font-size: 100px;
            color: white;
            font-weight: bold;
        }

        .header {
            text-align: center;
            color: white;
            font-size: 40px;
            font-weight: bold;
        }

        .subheader {
            color: #d1ccd9;
            font-size: 20px;
        }

        # .button {
        #     background-color:#32cdc8 ;
        #     color: #408291;
        #     font-weight: bold;
        #     padding: 10px 20px;  /* Padding for size */
        #     border: none;  /* Remove border */
        #     border-radius: 12px;  /* Round corners */
        #     cursor: pointer;  /* Add pointer cursor on hover */
        #     transition: background-color 0.3s ease;
        # }
        .stButton>button {
            background-color: white; /* Button background color */
            color: #30b6a1 ;  /* Text color */
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #32cdc8; /* Darker background color on hover */
        }

        .stButton>button:focus {
            outline: none;  /* Remove outline when focused */
        }
        .prediction {
            font-size: 20px;
            color: #ff1493;
            font-weight: bold;
        }
        .css-1lcbmhc {
            background-color: rgba(255, 255, 255, 0); /* Remove white background */
        }
    </style>
""", unsafe_allow_html=True)

# Add your content
st.markdown('<div class="title">Weather Prediction</div>', unsafe_allow_html=True)
st.markdown('<div style= "font-size: 40px; font-weight: bold; color: white">Predicting Maximum Temperature</div>', unsafe_allow_html=True)

# Button to predict temperature
predict_button = st.button("Predict Temperature")

if predict_button:
    # Display the prediction result when button is pressed
    st.markdown('<div class="prediction">Prediction: 28°C</div>', unsafe_allow_html=True)

# Welcome Text
st.markdown('<div style="text-align: center; font-size: 40px; font-weight: bold; color: white;">Welcome to the Weather Prediction App!</div>', unsafe_allow_html=True)


st.markdown("""
    <div style="font-size:20px;">
        This app uses a <strong>Linear Regression</strong> model to predict the maximum temperature (`temp_max`) based on various weather features such as 
        precipitation, temperature, wind speed, and more. Explore the data and make your own predictions!
    </div>
""", unsafe_allow_html=True)
# EDA Section Title
st.markdown('<div class="header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)

# Display data sample
st.subheader("Sample Data")
st.write(data.head())

# Temperature Distribution plot
st.subheader("Temperature Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(data['temp_max'], bins=20, kde=True, color='skyblue')
plt.title("Max Temperature Distribution")
plt.xlabel("Temperature (°F)")
plt.ylabel("Frequency")
st.pyplot(plt)

# Feature correlation heatmap
st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(10, 6))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Features")
st.pyplot(plt)

# Scatter plot: Temperature vs. Weather Encoded
st.subheader("Max Temperature vs. Encoded Weather")
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data['weather_encoded'], y=data['temp_max'], color='green')
plt.title("Max Temperature vs Weather")
plt.xlabel("Weather (Encoded)")
plt.ylabel("Max Temperature (°F)")
st.pyplot(plt)

# Model Section

st.markdown('<div class="header">Model: Linear Regression </div>', unsafe_allow_html=True)
# Display Mean Squared Error
st.subheader("Model Performance")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")

# Display some sample predictions
st.subheader("Sample Predictions vs Actual Values")
prediction_df = pd.DataFrame({
    "Predicted": y_pred[:10],
    "Actual": y_test[:10].values
})
st.write(prediction_df)

# Interactive temperature prediction slider
st.markdown('<div class="header"> Predict Max Temperature for a Given Day</div>', unsafe_allow_html=True)


# Let user choose a feature value using sliders
year = st.slider("Year", int(data['year'].min()), int(data['year'].max()), 2015)
month = st.slider("Month", 1, 12, 6)
day = st.slider("Day", 1, 31, 15)
weekday = st.slider("Weekday", 0, 6, 2)  # 0 = Monday, 6 = Sunday

# Make sure input_data has the same columns and order as X_train
input_data = pd.DataFrame({
    'year': [year],
    'month': [month],
    'day': [day],
    'weekday': [weekday],
    'precipitation': [1.2],  # Example value
    'temp_min': [14],         # Example value
    'wind': [5],              # Example value
    'weather_encoded': [0]    # Example encoded value (adjust as needed)
})

# Ensure the column order matches the model's input
input_data = input_data[X_train.columns]

# Predict with the same column order
predicted_temp = model.predict(input_data)

# Show the predicted temperature
st.markdown(f'<p class="prediction">Predicted Max Temperature: {predicted_temp[0]:.2f} °F</p>', unsafe_allow_html=True)


# Conclusion Section Title
st.markdown('<div class="header">Conclusion</div>', unsafe_allow_html=True)

# Conclusion Text
st.markdown("""
    <div style="font-size:20px;">
        In this project, we developed a linear regression model to predict the maximum temperature based on various weather features.

**Key Insights:**
- The correlation heatmap shows the relationships between different features, which helped in selecting the relevant features for our model.
- The model's performance, as measured by Mean Squared Error (MSE), provides an indication of its prediction accuracy.
- The interactive widget allows users to input various features and predict the maximum temperature for a given day.
    </div>
""", unsafe_allow_html=True)




