import pandas as pd
import pickle as pc

new_data = pd.DataFrame([[57811, 57900, 57700]], columns=['Open', 'High', 'Low'])

# Load the Model
with open('Multilinear_regression_model.pkl', 'rb') as file:
    Multilinear_model = pc.load(file)

# Load the Model
with open('decision_tree_model.pkl', 'rb') as file:
    Tree_model = pc.load(file)
   

print("LinearModel: ",Multilinear_model.predict(new_data))
print("TreeModel: ",Tree_model.predict(new_data))