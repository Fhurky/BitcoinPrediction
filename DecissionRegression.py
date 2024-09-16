import pandas as pd
import pickle as pc
from sklearn.tree import DecisionTreeRegressor
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

# Separate the target variable (Y) and features (X)
Y = data['Close']  # Define 'Close' column as the target variable
X = data.drop(['Close'], axis=1)  # Drop 'Close' column from features

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)

# Initialize and fit the DecisionTreeRegressor model
Tree = DecisionTreeRegressor(criterion='squared_error')
Tree.fit(train_x, train_y)  # Fit the model to the training data

# Save the model
try:
    with open('decision_tree_model.pkl', 'wb') as file:
        pc.dump(Tree, file)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

# Make predictions using the test set
predictions = Tree.predict(test_x)

# Evaluate and print the R2 score of the model
print("R2-score: %.2f" % r2_score(test_y, predictions))

# Define a function for making predictions with new data
def prediction(unknown):
    """
    Predict the target variable for new data using the saved model.
    
    Parameters:
    unknown (pd.DataFrame): New data for which predictions are to be made.
    
    Returns:
    numpy.ndarray: Predicted values.
    """
    if not isinstance(unknown, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if unknown.shape[1] != train_x.shape[1]:
        raise ValueError("Input data must have the same number of features as the training data")
    
    # Load the saved model
    try:
        with open('decision_tree_model.pkl', 'rb') as file:
            model = pc.load(file)
        return model.predict(unknown)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None