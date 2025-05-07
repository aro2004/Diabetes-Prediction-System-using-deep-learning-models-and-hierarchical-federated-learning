import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models
import tensorflow as tf  # Import TensorFlow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import rsa
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ===-------------------------= INPUT DATA --------------------
dataframe = pd.read_csv("/content/diabetes_dataset.csv")

# === Basic Data Analysis ===
print("Dataset Information:")
print(dataframe.info())

# Check the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(dataframe.head())

# === Handle Missing Values ===
print("\nHandling Missing Values")
print(dataframe.isnull().sum())
if dataframe.isnull().sum().any():
    dataframe = dataframe.dropna()

# === Summary Statistics ===
print("\nSummary Statistics:")
print(dataframe.describe())

# === Correlation Map ===
plt.figure(figsize=(10, 8))
correlation_matrix = dataframe.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()

# === Scatter Plots ===
plt.figure(figsize=(12, 8))

# Scatter plot for 'Glucose' vs. 'BMI' vs 'Outcome'
plt.subplot(2, 2, 1)
sns.scatterplot(data=dataframe, x='Glucose', y='BMI', hue='Outcome', palette='viridis')
plt.title("Glucose vs BMI vs Outcome")

# Scatter plot for 'Pregnancies' vs. 'Age' vs 'Outcome'
plt.subplot(2, 2, 2)
sns.scatterplot(data=dataframe, x='Pregnancies', y='Age', hue='Outcome', palette='viridis')
plt.title("Pregnancies vs Age vs Outcome")

# Scatter plot for 'Insulin' vs 'SkinThickness' vs 'Outcome'
plt.subplot(2, 2, 3)
sns.scatterplot(data=dataframe, x='Insulin', y='SkinThickness', hue='Outcome', palette='viridis')
plt.title("Insulin vs SkinThickness vs Outcome")

# Scatter plot for 'BloodPressure' vs 'BMI' vs 'Outcome'
plt.subplot(2, 2, 4)
sns.scatterplot(data=dataframe, x='BloodPressure', y='BMI', hue='Outcome', palette='viridis')
plt.title("BloodPressure vs BMI vs Outcome")

plt.tight_layout()
plt.show()

# ================== DATA SPLITTING  ====================
X = dataframe.drop('Outcome', axis=1)
y = dataframe['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize the data for deep learning models
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================== CAPSULE NETWORK ====================
def capsule_network(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with a single neuron and sigmoid activation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

capsule_model = capsule_network((X_train.shape[1],))

# ================== RADIAL BASIS FUNCTION NETWORK (RBFN) ====================
def rbf_network(X_train, y_train):
    model = SVC(kernel='rbf', random_state=0)
    model.fit(X_train, y_train)  # No epochs needed for SVM
    return model

# ================== ATTENTION MECHANISM ====================
def attention_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    attention = layers.Attention()([inputs, inputs])
    x = layers.GlobalAveragePooling1D()(attention)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)  # Output layer with a single neuron and sigmoid activation
    model = models.Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

attention_model_instance = attention_model((X_train.shape[1], 1))
X_train_attention = np.expand_dims(X_train, axis=-1)
X_test_attention = np.expand_dims(X_test, axis=-1)

# ================== HIERARCHICAL FEDERATED LEARNING ====================
num_clients = 5
client_data = []
client_labels = []

# Split data among clients
for i in range(num_clients):
    X_client, y_client = X_train[i::num_clients], y_train[i::num_clients]
    client_data.append(X_client)
    client_labels.append(y_client)

# Training function for local models
def train_client(model, X_client, y_client):
    model.fit(X_client, y_client, epochs=5, batch_size=32, verbose=0)
    return model

# Train models for each client using different models (Capsule, RBF, Attention)
def train_federated_models():
    client_models = []
    
    for i in range(num_clients):
        if i % 3 == 0:
            model = capsule_network((X_train.shape[1],))
        elif i % 3 == 1:
            model = rbf_network(X_train, y_train)  # No epochs here for SVM
        else:
            model = attention_model((X_train.shape[1], 1))

        # If it’s not an SVM model, train it with epochs
        if isinstance(model, models.Model):
            trained_model = train_client(model, client_data[i], client_labels[i])
            client_models.append(trained_model)
        else:
            client_models.append(model)  # Add the SVM model without training in Keras style
    
    return client_models

client_models = train_federated_models()

# Federated Averaging: Aggregate models at local server level
def federated_averaging(models):
    global_model = capsule_network((X_train.shape[1],))  # Use Capsule model as a reference for aggregation
    X_train_combined = np.vstack([client_data[i] for i in range(len(client_data))])
    y_train_combined = np.hstack([client_labels[i] for i in range(len(client_labels))])
    global_model.fit(X_train_combined, y_train_combined, epochs=5, batch_size=32, verbose=0)
    return global_model

# Hierarchical Aggregation: Aggregate models at the global server level
def hierarchical_aggregation(local_models):
    return federated_averaging(local_models)

# Split client models into two groups for hierarchical aggregation
local_server_1 = client_models[:3]
local_server_2 = client_models[3:]

local_model_1 = hierarchical_aggregation(local_server_1)
local_model_2 = hierarchical_aggregation(local_server_2)

# Final global model aggregation
final_global_model = federated_averaging([local_model_1, local_model_2])

# Evaluate the final global model
final_predictions_global = final_global_model.predict(X_test)
accuracy_global = np.mean((final_predictions_global > 0.5).flatten() == y_test) * 100
print(f"Final Global Model Accuracy (Hierarchical Federated Learning): {accuracy_global}%")
print(classification_report(y_test, (final_predictions_global > 0.5)))

# =================== PREDICTION ====================
print("Prediction - Diabetic")
a1 = int(input("Enter Pregnancies = "))
a2 = int(input("Enter Glucose Level = "))
a3 = int(input("Enter Blood Pressure Level = "))
a4 = int(input("Enter SkinThickness Level = "))
a5 = int(input("Enter Insulin Level = "))
a6 = float(input("Enter BMI Level = "))
a7 = float(input("Enter DiabetesPedigreeFunction Level = "))
a8 = int(input("Enter Age = "))

Data_reg = [a1, a2, a3, a4, a5, a6, a7, a8]
Data_reg_scaled = scaler.transform([Data_reg])

# Use Final Global Model for Prediction
prediction = final_global_model.predict(Data_reg_scaled)
print(f"Global Model Prediction: {'DIABETES' if prediction > 0.5 else 'NOT AFFECTED'}")

# ================== ENCRYPTION AND DECRYPTION ====================
# RSA Encryption for predictions
prediction = "DIABETES" if prediction > 0.5 else "NOT AFFECTED"
public_key, private_key = rsa.newkeys(512)

# Encrypt the prediction
prediction_bytes = prediction.encode('utf-8')
encrypted_prediction = rsa.encrypt(prediction_bytes, public_key)

# Save the encrypted prediction
save_path = 'Cloud/Encrypt'
os.makedirs(save_path, exist_ok=True)
file_path = os.path.join(save_path, "Encrypted_Prediction.txt")
with open(file_path, "wb") as f:
    f.write(encrypted_prediction)

# Decrypt the prediction
decrypted_prediction_bytes = rsa.decrypt(encrypted_prediction, private_key)
decrypted_prediction = decrypted_prediction_bytes.decode('utf-8')

print(f"Encrypted Data: {encrypted_prediction}")
print(f"Decrypted Data: {decrypted_prediction}")

# Save prediction result
with open('prediction.pickle', 'wb') as f:
    pickle.dump(prediction, f)

# ================== ENCRYPTION OF MODELS ====================
def encrypt_model(model, public_key, save_path):
    model_bytes = pickle.dumps(model)
    aes_key = get_random_bytes(32)
    cipher_aes = AES.new(aes_key, AES.MODE_GCM)
    ciphertext, tag = cipher_aes.encrypt_and_digest(model_bytes)
    encrypted_aes_key = rsa.encrypt(aes_key, public_key)
    model_file_path = os.path.join(save_path, "Encrypted_Model_AES_RSA.pkl")
    with open(model_file_path, 'wb') as file:
        file.write(encrypted_aes_key)
        file.write(cipher_aes.nonce)
        file.write(tag)
        file.write(ciphertext)

    print(f"Model encrypted and saved at {model_file_path}")

encrypt_model(final_global_model, public_key, 'Cloud/Encrypt')
