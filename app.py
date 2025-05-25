from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# Initialize model and scaler as None
classifier = None
scaler = None

def load_model():
    """Load or train the diabetes prediction model"""
    global classifier, scaler
    
    # Load dataset
    diabetes_dataset = pd.read_csv('diabetes.csv')
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    
    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        standardized_data, Y, test_size=0.2, stratify=Y, random_state=2
    )
    
    # Train model
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    
    # Print accuracy (for debugging)
    train_acc = accuracy_score(classifier.predict(X_train), Y_train)
    test_acc = accuracy_score(classifier.predict(X_test), Y_test)
    print(f"Model trained - Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

# Load model when starting (Vercel runs this on cold start)
load_model()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if classifier is None or scaler is None:
            load_model()  # Ensure model is loaded
        
        # Get form data
        data = {
            'Pregnancies': float(request.form.get('Pregnancies')),
            'Glucose': float(request.form.get('Glucose')),
            'BloodPressure': float(request.form.get('BloodPressure')),
            'SkinThickness': float(request.form.get('SkinThickness')),
            'Insulin': float(request.form.get('Insulin')),
            'BMI': float(request.form.get('BMI')),
            'DiabetesPedigreeFunction': float(request.form.get('DiabetesPedigreeFunction')),
            'Age': float(request.form.get('Age'))
        }
        
        # Prepare input array
        input_data = np.array([
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]).reshape(1, -1)
        
        # Standardize and predict
        std_data = scaler.transform(input_data)
        prediction = classifier.predict(std_data)
        
        # Return result
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return render_template(
            'index.html',
            prediction_text=f'Prediction: {result}',
            form_data=data
        )
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            form_data=request.form if request.method == 'POST' else None
        )

# For Vercel deployment
@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return send_from_directory(os.path.join(root_dir, 'static'), path)

if __name__ == '__main__':
    app.run(debug=True)