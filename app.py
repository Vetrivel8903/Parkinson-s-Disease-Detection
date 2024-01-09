import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

# Ensure the 'uploads' folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the dataset
df = pd.read_csv("//Download the parkison data from online and paste that path directory here//")

# Extract numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
features = df[numeric_columns].values
labels = df['status'].values
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=1,
    grow_policy='depthwise',
    importance_type='gain',
    learning_rate=0.1,
    max_bin=256,
    max_cat_threshold=32,
    max_cat_to_onehot=4,
    max_delta_step=0,
    max_depth=3,
    max_leaves=0,
    min_child_weight=1,
    n_estimators=100,
    num_parallel_tree=1,
    random_state=42,
)

model.fit(x_train, y_train)

# Homepage route
@app.route('/')
def index():
    # Provide default feature values if feature_values is not defined
    feature_values = [0] * len(numeric_columns)

    # Calculate accuracy on the validation set
   # val_predictions = model.predict(x_val)
   # accuracy = accuracy_score(y_val, val_predictions) * 100

    return render_template('index.html', feature_values=feature_values)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is in the request
    if 'dataset' in request.files:
        file = request.files['dataset']

        # Check if the file has a valid extension
        if file and file.filename.endswith(('.csv', '.data')):
            # Save the uploaded file to the uploads folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)

            # Process the uploaded file as needed
            uploaded_df = pd.read_csv(file_path)

            # Extract numeric columns
            uploaded_numeric_columns = uploaded_df.select_dtypes(include=[np.number]).columns
            uploaded_features = uploaded_df[uploaded_numeric_columns].values

            # Ensure the input data has the correct shape
            if uploaded_features.shape[1] != x_train.shape[1]:
                return render_template('index.html', error_message="Feature shape mismatch. Please check your input data.")

            # Scale the input features
            input_data_scaled = scaler.transform(uploaded_features)

            # Calculate accuracy on the validation set
            val_predictions = model.predict(x_val)
            accuracy = accuracy_score(y_val, val_predictions) * 100

            # Make prediction
            prediction = model.predict(input_data_scaled)

            # Interpret the prediction
            if prediction[0] == 1:
                result_message = " PARKINSON'S DISEASE DETECTED."
            else:
                result_message = "NO PARKINSON'S DISEASE DETECTED."
            
            return render_template('result.html', prediction=prediction[0], feature_values=input_data_scaled.flatten().tolist(), result_message=result_message, accuracy=accuracy)

    return render_template('index.html')  # Redirect to the homepage if no file is uploaded

# New route to get file values
@app.route('/get_file_values', methods=['POST'])
def get_file_values():
    if 'dataset' in request.files:
        file = request.files['dataset']

        if file and file.filename.endswith(('.csv', '.data')):
            uploaded_df = pd.read_csv(file)
            uploaded_numeric_columns = uploaded_df.select_dtypes(include=[np.number]).columns
            uploaded_features = uploaded_df[uploaded_numeric_columns].values

            if uploaded_features.shape[1] != x_train.shape[1]:
                return jsonify({'error': 'Feature shape mismatch. Please check your input data.'}), 400

            input_data_scaled = scaler.transform(uploaded_features)

            return jsonify({'feature_values': input_data_scaled.flatten().tolist()})

    return jsonify({'error': 'File not uploaded'}), 400
    
if __name__ == '__main__':
    app.run(debug=True)
