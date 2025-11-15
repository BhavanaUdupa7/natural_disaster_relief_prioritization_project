#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset
df = pd.read_csv("disaster_data.csv")  # Replace with the actual filename if different

# Display the first few rows
df.head()


# In[3]:


# Drop unnecessary columns for prioritizing regions
df_clean = df[['Year', 'Disaster Type', 'Country', 'Region', 'Total Deaths', 'Total Affected', "Total Damages ('000 US$)"]]

# Convert numeric columns
df_clean['Total Deaths'] = pd.to_numeric(df_clean['Total Deaths'], errors='coerce')
df_clean['Total Affected'] = pd.to_numeric(df_clean['Total Affected'], errors='coerce')
df_clean["Total Damages ('000 US$)"] = pd.to_numeric(df_clean["Total Damages ('000 US$)"], errors='coerce')

# Drop rows with no country info
df_clean = df_clean.dropna(subset=['Country'])
df_clean.head()


# In[4]:


deaths_by_country = df_clean.groupby('Country')['Total Deaths'].sum().sort_values(ascending=False).head(10)
deaths_by_country


# In[5]:


affected_by_country = df_clean.groupby('Country')['Total Affected'].sum().sort_values(ascending=False).head(10)
affected_by_country


# In[6]:


damage_by_country = df_clean.groupby('Country')["Total Damages ('000 US$)"].sum().sort_values(ascending=False).head(10)
damage_by_country


# In[7]:


import matplotlib.pyplot as plt

# Deaths
deaths_by_country.plot(kind='bar', title='Top 10 Countries by Deaths from Natural Disasters', figsize=(10,5))
plt.ylabel("Total Deaths")
plt.show()

# Affected
affected_by_country.plot(kind='bar', title='Top 10 Countries by People Affected', figsize=(10,5), color='orange')
plt.ylabel("Total Affected")
plt.show()

# Damages
damage_by_country.plot(kind='bar', title='Top 10 Countries by Economic Damage (in 000 USD)', figsize=(10,5), color='green')
plt.ylabel("Total Damages ('000 US$)")
plt.show()


# In[8]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fill missing values
df_clean.fillna(0, inplace=True)

# Create impact score and label
impact_score = df_clean['Total Deaths'] + df_clean['Total Affected'] + df_clean["Total Damages ('000 US$)"]
df_clean['Impact Level'] = pd.qcut(impact_score, q=3, labels=['Low', 'Medium', 'High'])

# Encode categorical variables
le_country = LabelEncoder()
df_clean['Country_enc'] = le_country.fit_transform(df_clean['Country'])

le_disaster = LabelEncoder()
df_clean['Disaster Type_enc'] = le_disaster.fit_transform(df_clean['Disaster Type'])

# Select features and label
features = df_clean[['Year', 'Country_enc', 'Disaster Type_enc', 'Total Deaths', 'Total Affected', "Total Damages ('000 US$)"]]
labels = df_clean['Impact Level']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[10]:


from sklearn.ensemble import RandomForestClassifier

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[11]:


# Predict on test data
y_pred = rf_model.predict(X_test)


# In[12]:


# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()


# In[14]:


# Feature names
feature_names = ['Year', 'Country_enc', 'Disaster Type_enc', 'Total Deaths', 'Total Affected', "Total Damages ('000 US$)"]

# Get feature importances
importances = rf_model.feature_importances_

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# In[15]:


import numpy as np

# Example input (replace with real encoded values)
# Format: (Year, Country_enc, Disaster Type_enc, Total Deaths, Total Affected, Damages)
input_data = (2022, 100, 5, 200, 10000, 5000)

# Convert and reshape input
input_data_np = np.asarray(input_data).reshape(1, -1)

# Predict
prediction = rf_model.predict(input_data_np)
print("Predicted Impact Level:", prediction[0])

# Interpret result
if prediction[0] == 'Low':
    print("This disaster is predicted to have LOW impact.")
elif prediction[0] == 'Medium':
    print("This disaster is predicted to have MEDIUM impact.")
else:
    print("This disaster is predicted to have HIGH impact.")


# In[16]:


import joblib

# Save your trained Random Forest model
joblib.dump(rf_model, 'disaster_impact_model.pkl')

# Save encoders if needed
joblib.dump(le_country, 'country_encoder.pkl')
joblib.dump(le_disaster, 'disaster_encoder.pkl')


# In[18]:


from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('disaster_impact_model.pkl')
le_country = joblib.load('country_encoder.pkl')
le_disaster = joblib.load('disaster_encoder.pkl')

@app.route('/predict-impact', methods=['POST'])
def predict_impact():
    data = request.get_json()
    
    # You may decode/encode strings here if needed
    # For now, expecting numeric encoded inputs
    
    input_data = np.array([[
        data['year'],
        data['country_enc'],
        data['disaster_type_enc'],
        data['total_deaths'],
        data['total_affected'],
        data['total_damages']
    ]])

    prediction = model.predict(input_data)[0]

    return jsonify({'predicted_impact': prediction})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




