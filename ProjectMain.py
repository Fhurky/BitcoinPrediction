import pandas as pd
import numpy as np
import pickle as pc
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# CSV dosyasını okuma
data = pd.read_csv("btcusd.csv")

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
print(f"LSTM Model Tahmini: {lstm_prediction:.2f}")
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
print("LinearModel: ", linear_predictions)
print("TreeModel: ", tree_predictions)
print("ForestModel: ", forest_predictions)
print("SVRModel: ", svr_predictions)
print("KNNModel: ", knn_predictions)
print("LSTMModel: ", lstm_prediction)

# Modellerin isimleri ve tahmin sonuçlarının ortalamaları
model_names = ['Linear Model', 'Tree Model', 'Forest Model', 'SVR Model', 'KNN Model', 'LSTM Model']
predictions_means = [
    np.mean(linear_predictions),
    np.mean(tree_predictions),
    np.mean(forest_predictions),
    np.mean(svr_predictions),
    np.mean(knn_predictions),
    np.mean(lstm_prediction),
]

# Tüm modellerin tahmin ortalamasını hesapla
average_mean = np.mean(predictions_means)

# Bar grafiği oluşturma
plt.figure(figsize=(12, 7))

# Her modelin tahmin ortalamasını çubuk olarak göster
plt.bar(model_names, predictions_means, color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'], label='Model Averages')

# Tüm modellerin ortalamasını ek bir çubuk olarak göster
plt.bar('Average of Models', average_mean, color='gray', label='Average of Models')

# Başlık ve etiketler
plt.title('Model Prediction Averages and Overall Model Average')
plt.xlabel('Models')
plt.ylabel('Prediction Average')

# Y ekseni limitlerini ayarla
plt.ylim([min(predictions_means + [average_mean]) - 20, max(predictions_means + [average_mean]) + 20])

plt.xticks(rotation=45)  # X ekseni etiketlerini döndür
plt.legend()
plt.show()
