import pandas as pd
import pickle as pc
import matplotlib.pyplot as plt
import numpy as np

new_data = pd.DataFrame([[57864, 57864, 57836]], columns=['Open', 'High', 'Low'])

# Load the Model
with open('Multilinear_regression_model.pkl', 'rb') as file:
    Multilinear_model = pc.load(file)

# Load the Model
with open('decision_tree_model.pkl', 'rb') as file:
    Tree_model = pc.load(file)

# Load the Model
with open('RandomForest_regression_model.pkl', 'rb') as file:
    Forest_model = pc.load(file)

# Load the Model
with open('SVR_regression_model.pkl', 'rb') as file:
    SVR_model = pc.load(file)

# Load the Model
with open('KNN_regression_model.pkl', 'rb') as file:
    KNN_model = pc.load(file)

linear_predictions = Multilinear_model.predict(new_data)
tree_predictions = Tree_model.predict(new_data)
forest_predictions = Forest_model.predict(new_data)
svr_predictions = SVR_model.predict(new_data)
knn_predictions = KNN_model.predict(new_data)
   
print("LinearModel: ",linear_predictions)
print("TreeModel: ",tree_predictions)
print("ForestModel: ",forest_predictions)
print("SvrModel: ",svr_predictions)
print("KnnModel: ",knn_predictions)

# Modellerin isimleri ve tahmin sonuçlarının ortalamaları
model_names = ['Linear Model', 'Tree Model', 'Forest Model', 'SVR Model', 'KNN Model']
predictions_means = [
    np.mean(linear_predictions),
    np.mean(tree_predictions),
    np.mean(forest_predictions),
    np.mean(svr_predictions),
    np.mean(knn_predictions)
]

# Calculate the average of all models' predictions
average_mean = np.mean(predictions_means)

# Create a bar chart
plt.figure(figsize=(12, 7))

# Plot each model's average prediction as a bar
plt.bar(model_names, predictions_means, color=['blue', 'green', 'orange', 'red', 'purple'], label='Model Averages')

# Plot the average of all models as an additional bar
plt.bar('Average of Models', average_mean, color='gray', label='Average of Models')

# Title and labels
plt.title('Model Prediction Averages and Overall Model Average')
plt.xlabel('Models')
plt.ylabel('Prediction Average')

# Set y-axis limit to show a range of 100 units
plt.ylim([min(predictions_means + [average_mean]) - 20, max(predictions_means + [average_mean]) + 20])

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.show()