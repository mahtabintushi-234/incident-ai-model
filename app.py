import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, request, jsonify
import joblib


# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic features for system metrics (e.g., CPU usage, memory usage)
data = {
    'cpu_usage': np.random.uniform(30, 100, 1000),  # CPU usage from 30% to 100%
    'memory_usage': np.random.uniform(50, 100, 1000),  # Memory usage from 50% to 100%
    'error_count': np.random.randint(0, 10, 1000),  # Random error counts between 0 and 10
    'response_time': np.random.uniform(100, 1000, 1000),  # Response times between 100ms and 1000ms
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define thresholds to classify incidents
# If CPU usage is over 85% and/or error count > 5, it's critical; else, normal or warning
df['incident'] = np.where((df['cpu_usage'] > 85) & (df['error_count'] > 5), 'critical', 'normal')
df['incident'] = np.where((df['cpu_usage'] > 70) & (df['error_count'] > 3) & (df['incident'] != 'critical'), 'warning', df['incident'])

# Encode the labels as numeric values for machine learning
label_encoder = LabelEncoder()
df['incident'] = label_encoder.fit_transform(df['incident'])  # critical = 2, warning = 1, normal = 0

# Display a sample of the data
print(df.head())

# Scale the features (standardize them to mean 0, variance 1)
scaler = StandardScaler()
df[['cpu_usage', 'memory_usage', 'error_count', 'response_time']] = scaler.fit_transform(df[['cpu_usage', 'memory_usage', 'error_count', 'response_time']])

# Split data into features (X) and labels (y)
X = df[['cpu_usage', 'memory_usage', 'error_count', 'response_time']]
y = df['incident']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, 'incident_model.pkl')

app = Flask(__name__)

# Load the trained model
model = joblib.load('incident_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_incident():
    # Get data from the POST request
    data = request.get_json()

    # Extract features from the request
    features = np.array([data['cpu_usage'], data['memory_usage'], data['error_count'], data['response_time']]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(features)

    # Convert numeric prediction to label (normal, warning, critical)
    labels = {0: 'normal', 1: 'warning', 2: 'critical'}
    return jsonify({'incident': labels[prediction[0]]})

if __name__ == '__main__':
    app.run(debug=True)
    
joblib.dump(model, 'incident_model.pkl')


