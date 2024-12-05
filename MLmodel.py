
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from google.colab import files

# Step 1: Upload the dataset
print("Please upload the dataset (CSV format).")
uploaded = files.upload()

# Step 2: Load the dataset
data = pd.read_csv(list(uploaded.keys())[0])  # Automatically load the uploaded file
print("Dataset loaded successfully!")
print("Preview of dataset:")
print(data.head())

# Step 3: Encode categorical columns
label_encoders = {}
for column in ['state', 'soil_type', 'crop', 'Sustainable Farming Method', 'season']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Step 4: Define features (inputs) and targets (outputs)
features = data[['month', 'state', 'soil_type']]
targets = data[['crop', 'Sustainable Farming Method', 'season']]

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Step 6: Initialize the model
rf = RandomForestClassifier(random_state=42, n_estimators=100)
model = MultiOutputClassifier(rf)

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Predict on the test set and calculate accuracy
y_pred = model.predict(X_test)
crop_accuracy = accuracy_score(y_test['crop'], y_pred[:, 0])
farming_method_accuracy = accuracy_score(y_test['Sustainable Farming Method'], y_pred[:, 1])
season_accuracy = accuracy_score(y_test['season'], y_pred[:, 2])

print(f'Crop Prediction Accuracy: {crop_accuracy:.2f}')
print(f'Farming Method Prediction Accuracy: {farming_method_accuracy:.2f}')
print(f'Season Prediction Accuracy: {season_accuracy:.2f}')

# Step 9: User inputs for predictions
print("\nEnter the details for prediction:")
user_month = int(input("Enter the month (as a number, e.g., 5 for May): "))
user_state = input("Enter the state (e.g., Telangana): ")
user_soil_type = input("Enter the soil type (e.g., Clay): ")

# Step 10: Encode user inputs
state_encoded = label_encoders['state'].transform([user_state])[0]
soil_type_encoded = label_encoders['soil_type'].transform([user_soil_type])[0]

# Prepare input data
new_input = pd.DataFrame({
    'month': [user_month],
    'state': [state_encoded],
    'soil_type': [soil_type_encoded]
})

# Step 11: Make predictions
predictions = model.predict(new_input)
predicted_crop = label_encoders['crop'].inverse_transform([predictions[0][0]])[0]
predicted_method = label_encoders['Sustainable Farming Method'].inverse_transform([predictions[0][1]])[0]
predicted_season = label_encoders['season'].inverse_transform([predictions[0][2]])[0]

# Step 12: Display predictions
print(f"\nPrediction Results:")
print(f"Predicted Sustainable Crop: {predicted_crop}")
print(f"Predicted Sustainable Farming Method: {predicted_method}")
print(f"Predicted Season: {predicted_season}")
