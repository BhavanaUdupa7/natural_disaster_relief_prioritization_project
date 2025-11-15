# Natural Disaster Relief Prioritization Project

This project predicts the **impact level of natural disasters** (Low, Medium, High) using **Machine Learning** and provides a simple **Flask API** to get predictions.  
It helps prioritize disaster relief and resource allocation based on severity.

## How It Works

- Cleans and processes disaster data (`disaster_data.csv`)
- Trains a **Random Forest Classifier** using features like deaths, affected people, and damages
- Encodes categorical values such as country and disaster type
- Saves the trained model and encoders for reuse
- Runs a Flask API that predicts the disaster’s **impact level**

## Tech Stack

- **Python**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Flask**, **Flask-CORS**
- **Joblib**

## Run Instructions
Run the index file
