import pandas as pd
import pickle as pc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
print(os.getcwd())


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

# Separate the target variable (Y) and features (X)
Y = data['Close']  # Define 'Close' column as the target variable
X = data.drop(['Close'], axis=1)  # Drop 'Close' column from features

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

# Create and train the Linear Regression model
LM = LinearRegression()
LM.fit(train_x, train_y)

# Save the Model
try:
    with open('Multilinear_regression_model.pkl', 'wb') as file:
        pc.dump(LM, file)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

# Make predictions using the test set
predictions = LM.predict(test_x)

# Evaluate and print the R2 score of the model
print("R2-score: %.2f" % r2_score(test_y, predictions))

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
    
    return LM.predict(unknown)
