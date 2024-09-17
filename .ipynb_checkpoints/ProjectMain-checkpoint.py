import pandas as pd
import numpy as np
import pickle as pc
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import configparser

# CSV dosyasını okuma
data = pd.read_csv("Data/btcusd.csv")

# Config dosyasını oluşturma ve okuma
config = configparser.ConfigParser()
config.read('application.properties')

# Verinin ters çevrilmiş hali
rows_1_to_50 = data.iloc[1:26][::-1]  # Veriyi ters çevirme
test4LSTM = rows_1_to_50['Close'].values

# Yeni veriyi DataFrame'e dönüştürme
new_data = pd.DataFrame(test4LSTM, columns=['Close'])

# 'Close' verilerini normalizasyon için fit edilmiştir
scaler = MinMaxScaler()
scaler.fit(data[['Close']])  # Eğitim sırasında kullanılan scaler'ı fit et

# Veriyi normalizasyon için ölçekleme
test4LSTM_scaled = scaler.transform(new_data)

# LSTM için uygun formatta veri hazırlama
# TIME_STEPS, LSTM modelinizi eğitirken kullandığınız zaman adımı sayısıdır
TIME_STEPS = 25
X_test_seq = np.array(test4LSTM_scaled).reshape(1, TIME_STEPS, 1)  # (1, 25, 1) şeklinde

# LSTM modelini yükleme
lstm_model = load_model('Models/LSTM_model_close_only.h5')

# Tahmin yapma
lstm_prediction_scaled = lstm_model.predict(X_test_seq)

# Tahmini geri ölçeklendirme
lstm_prediction = scaler.inverse_transform(lstm_prediction_scaled.reshape(-1, 1))[0][0]

# Tahmini gösterme
# print(f"LSTM Model Tahmini: {lstm_prediction:.2f}")

# Diğer modellerle tahmin yapma
with open('Models/Multilinear_regression_model.pkl', 'rb') as file:
    Multilinear_model = pc.load(file)

# Load the Model
with open('Models/decision_tree_model.pkl', 'rb') as file:
    Tree_model = pc.load(file)

# Load the Model
with open('Models/RandomForest_regression_model.pkl', 'rb') as file:
    Forest_model = pc.load(file)

# Load the Model
with open('Models/SVR_regression_model.pkl', 'rb') as file:
    SVR_model = pc.load(file)

# Load the Model
with open('Models/KNN_regression_model.pkl', 'rb') as file:
    KNN_model = pc.load(file)

# Verileri doğru formata dönüştürme
new_data_for_other_models = pd.DataFrame([[57854, 57864, 57835]], columns=['Open', 'High', 'Low'])

linear_predictions = Multilinear_model.predict(new_data_for_other_models)
tree_predictions = Tree_model.predict(new_data_for_other_models)
forest_predictions = Forest_model.predict(new_data_for_other_models)
svr_predictions = SVR_model.predict(new_data_for_other_models)
knn_predictions = KNN_model.predict(new_data_for_other_models)

# Tahminleri gösterme
# print("LinearModel: ", linear_predictions)
# print("TreeModel: ", tree_predictions)
# print("ForestModel: ", forest_predictions)
# print("SVRModel: ", svr_predictions)
# print("KNNModel: ", knn_predictions)
# print("LSTMModel: ", lstm_prediction)

# Mevcut değerleri okuma ve dönüşüm
MultilinearRegression_mse_score = float(config.get('Metrics', 'MultilinearRegression_MSE-score'))
decisionregression_mse_score = float(config.get('Metrics', 'decisionregression_mse-score'))
svrregression_mse_score = float(config.get('Metrics', 'svrregression_mse-score'))
lstmregression_mse_score = float(config.get('Metrics', 'lstmregression_mse-score'))
knnregression_mse_score = float(config.get('Metrics', 'knnregression_mse-score'))
forestregression_mse_score = float(config.get('Metrics', 'forestregression_mse-score'))

# Modellerin MSE skorları ve tahminleri
mse_scores = {
    'Multilinear Regression': MultilinearRegression_mse_score,
    'Decision Tree': decisionregression_mse_score,
    'SVR Regression': svrregression_mse_score,
    'LSTM Regression': lstmregression_mse_score,
    'KNN Regression': knnregression_mse_score,
    'Random Forest': forestregression_mse_score
}

predictions = {
    'Multilinear Regression': np.mean(linear_predictions),
    'Decision Tree': np.mean(tree_predictions),
    'SVR Regression': np.mean(svr_predictions),
    'LSTM Regression': lstm_prediction,
    'KNN Regression': np.mean(knn_predictions),
    'Random Forest': np.mean(forest_predictions)
}

# Hesaplama: İnvers MSE'ler ve ağırlıklar
inverse_mse = {model: 1 / mse for model, mse in mse_scores.items()}
total_inverse_mse = sum(inverse_mse.values())
weights = {model: inv_mse / total_inverse_mse for model, inv_mse in inverse_mse.items()}

# Ağırlıklı tahmini hesaplama
weighted_prediction = sum(weights[model] * predictions[model] for model in predictions)
# print(f"Weighted Prediction: {weighted_prediction:.2f}")

# Model isimleri ve tahminlerin ortalamaları
model_names = ['Multilinear Regression', 'Decision Tree', 'Random Forest', 'LSTM Regression', 'SVR Regression', 'KNN Model']
predictions_means = [
    np.mean(linear_predictions),
    np.mean(tree_predictions),
    np.mean(forest_predictions),
    lstm_prediction,
    np.mean(svr_predictions),
    np.mean(knn_predictions)
]

# Tüm modellerin tahmin ortalamasını hesapla
average_mean = np.mean(predictions_means)

# Bar grafiği oluşturma
plt.figure(figsize=(12, 7))

# Her modelin tahmin ortalamasını çubuk olarak göster
bars = plt.bar(model_names, predictions_means, color=['blue', 'green', 'orange', 'red', 'purple', 'magenta'], label='Model Averages')

# Tüm modellerin ortalamasını ek bir çubuk olarak göster
average_bar = plt.bar('Average of Models', average_mean, color='gray', label='Average of Models')

# Ağırlıklı tahmini gösteren çubuğu ekle
weighted_bar = plt.bar('Weighted Prediction', weighted_prediction, color='cyan', label='Weighted Prediction')

# Çubukların üstüne metin ekleme
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=10, color='black')

# Ortalama ve ağırlıklı tahmin çubuklarının üstüne metin ekleme
average_yval = average_bar[0].get_height()  # `average_bar`'ın içindeki çubukları al
weighted_yval = weighted_bar[0].get_height()  # `weighted_bar`'ın içindeki çubukları al
plt.text(average_bar[0].get_x() + average_bar[0].get_width()/2, average_yval, f'{average_yval:.2f}', va='bottom', ha='center', fontsize=10, color='black')
plt.text(weighted_bar[0].get_x() + weighted_bar[0].get_width()/2, weighted_yval, f'{weighted_yval:.2f}', va='bottom', ha='center', fontsize=10, color='black')

# Başlık ve etiketler
plt.title('Model Prediction Averages, Overall Model Average, and Weighted Prediction')
plt.xlabel('Models')
plt.ylabel('Prediction Average')

# Y ekseni limitlerini ayarla
plt.ylim([min(predictions_means + [average_mean, weighted_prediction]) - 5000, 
          max(predictions_means + [average_mean, weighted_prediction]) + 5000])

plt.xticks(rotation=45)  # X ekseni etiketlerini döndür
plt.legend()
plt.show()
