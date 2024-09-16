import pandas as pd
import pickle as pc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
data = data.sample(frac=0.1, random_state=0)

# Separate the target variable (Y) and features (X)
Y = data['Close']  # Define 'Close' column as the target variable
X = data.drop(['Close'], axis=1)  # Drop 'Close' column from features

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

RF_model = RandomForestRegressor(criterion='squared_error', n_jobs=-1,)
RF_model.fit(train_x, train_y)

# Save the Model
try:
    with open('RandomForest_regression_model.pkl', 'wb') as file:
        pc.dump(RF_model, file)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")


# Make predictions using the test set
predictions = RF_model.predict(test_x)

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
    
    return RF_model.predict(unknown)
