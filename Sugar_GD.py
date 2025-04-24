import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

# Baca dataset
df = pd.read_csv(r'C:\Users\Fernando\Documents\Telkom_University\S2 Teknik Elektro\Semester 2\Matematika Teknik Lanjut\UTS\sugar_consumption_dataset.csv')

# Tampilkan data awal
print("Data sebelum penghapusan duplikat:")
print(df.head())

# Filter Indonesia dan hapus duplikat
df_filtered = df[df['Country'] == 'Indonesia'] 
df_filtered_no_duplicates = df_filtered.drop_duplicates(subset='Total_Sugar_Consumption', keep='first')

# Rata-rata per tahun
df_avg_per_year = df_filtered_no_duplicates.groupby('Year')['Total_Sugar_Consumption'].mean().reset_index()

print("\nData setelah penghapusan duplikat (hanya Indonesia) dan merata-ratakan per tahun:")
print(df_avg_per_year.head())

# Siapkan data
X = df_avg_per_year['Year'].values.reshape(-1, 1)
y = df_avg_per_year['Total_Sugar_Consumption'].values

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Bangun model
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.0, clipnorm=1.0),
        loss='mean_squared_error'
    )
    return model

model = build_model()
model.fit(X_train, y_train, epochs=100, batch_size=len(X_train), verbose=0)  # Full-batch GD

# Evaluasi model
y_pred = model.predict(X_test)

if np.isnan(y_pred).any():
    print("❌ Model menghasilkan NaN pada prediksi!")
    print("Beberapa nilai y_pred:", y_pred[:5])
else:
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error pada data uji: {mse}")

# === Prediksi dan validasi tahun 2023 ===
year_to_predict = 2023

if year_to_predict in df_avg_per_year['Year'].values:
    actual_2023 = df_avg_per_year[df_avg_per_year['Year'] == year_to_predict]['Total_Sugar_Consumption'].values[0]
    
    year_scaled = scaler.transform([[year_to_predict]])
    prediction_2023 = model.predict(year_scaled)[0][0]
    
    print(f"\nPrediksi konsumsi gula tahun {year_to_predict}: {prediction_2023}")
    print(f"Nilai aktual tahun {year_to_predict}: {actual_2023}")
    print(f"Selisih prediksi dan aktual: {abs(prediction_2023 - actual_2023)}")
else:
    print(f"\nData tahun {year_to_predict} tidak tersedia di dataset untuk dibandingkan.")

# Visualisasi
plt.figure(figsize=(8, 6))
plt.scatter(df_avg_per_year['Year'], df_avg_per_year['Total_Sugar_Consumption'], color='blue', label='Data Indonesia')
plt.plot(
    df_avg_per_year['Year'],
    model.predict(scaler.transform(df_avg_per_year['Year'].values.reshape(-1, 1))),
    color='red', label='Prediksi Model'
)

plt.title('Plot Total Sugar Consumption untuk Indonesia dengan Model Neural Network (Gradient Descent)')
plt.xlabel('Tahun')
plt.ylabel('Total Sugar Consumption')
plt.legend()
plt.grid(True)
plt.show()
