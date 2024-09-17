import pandas as pd 
import pickle as pc
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load the data and drop unnecessary columns
try:
    data = pd.read_csv('btcusd.csv')
    data = data.drop(['Timestamp', 'Volume'], axis=1)  # Drop 'Timestamp' and 'Volume' columns
except FileNotFoundError:
    print("The file 'btcusd.csv' was not found.")
    exit()
except KeyError as e:
    print(f"Missing expected column in data: {e}")
    exit()

# Sample only 1% of the data
data = data.sample(frac=0.0001, random_state=0)

# Separate the target variable (Y) and features (X)
Y = data['Close']  # Define 'Close' column as the target variable
X = data.drop(['Close'], axis=1)  # Drop 'Close' column from features

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

SvrModel = SVR(kernel='linear')

start_time = time.time()

SvrModel.fit(train_x, train_y)

end_time = time.time()

training_time = end_time - start_time
print(f"Model training time: {training_time:.2f} seconds")

# Save the Model
try:
    with open('Models/SVR_regression_model.pkl', 'wb') as file:
        pc.dump(SvrModel, file)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

# Make predictions using the test set
predictions = SvrModel.predict(test_x)

# Evaluate and print the R2 score of the model
print("R2-score: %.2f" % r2_score(test_y, predictions))
print("MSE-score: %.2f" % mean_squared_error(test_y, predictions))

# Randomly sample 100 data points from the test set
sample_size = 100
sample_indices = np.random.choice(len(test_y), size=sample_size, replace=False)
sample_test_y = test_y.iloc[sample_indices]
sample_predictions = predictions[sample_indices]

# Plot the predicted vs actual values for the sampled data
plt.figure(figsize=(10, 6))
plt.scatter(sample_test_y, sample_predictions, color='blue', marker='o', alpha=0.5)
plt.plot([min(sample_test_y), max(sample_test_y)], [min(sample_test_y), max(sample_test_y)], color='red', linestyle='--', lw=2)
plt.title('Predicted vs Actual Values (Sampled Data)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

def r2_score():
    return  r2_score(test_y, predictions)

# Define a function for making predictions with new data
def prediction(unknown):
    """
    Predict the target variable for new data using the trained model.
    
    Parameters:
    unknown (pd.DataFrame): New data for which predictions are to be made.
    
    Returns:
    numpy.ndarray: Predicted values.
    """
    if not isinstance(unknown, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if unknown.shape[1] != train_x.shape[1]:
        raise ValueError("Input data must have the same number of features as the training data")
    
    return SvrModel.predict(unknown)