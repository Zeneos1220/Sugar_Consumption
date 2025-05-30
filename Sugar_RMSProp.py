import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'file.csv')

print("Data sebelum penghapusan duplikat:")
print(df.head())

df_filtered = df[df['Country'] == 'Indonesia']
df_filtered_no_duplicates = df_filtered.drop_duplicates(subset='Total_Sugar_Consumption', keep='first')

df_avg_per_year = df_filtered_no_duplicates.groupby('Year')['Total_Sugar_Consumption'].mean().reset_index()

print("\nData setelah penghapusan duplikat (hanya Indonesia) dan merata-ratakan per tahun:")
print(df_avg_per_year.head())

X = df_avg_per_year['Year'].values.reshape(-1, 1)
y = df_avg_per_year['Total_Sugar_Consumption'].values  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model

model = build_model()
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error pada data uji: {mse}")

year_to_predict = 2023

if year_to_predict in df_avg_per_year['Year'].values:
    actual_2023 = df_avg_per_year[df_avg_per_year['Year'] == year_to_predict]['Total_Sugar_Consumption'].values[0]
    
    year_2023_scaled = scaler.transform([[year_to_predict]])
    prediction_2023 = model.predict(year_2023_scaled)[0][0]
    
    print(f"\nPrediksi konsumsi gula tahun {year_to_predict}: {prediction_2023}")
    print(f"Nilai aktual konsumsi gula tahun {year_to_predict}: {actual_2023}")
    print(f"Selisih prediksi dan aktual tahun {year_to_predict}: {abs(prediction_2023 - actual_2023)}")
else:
    print(f"\nData untuk tahun {year_to_predict} tidak tersedia di dataset.")

year_next = max(df_avg_per_year['Year']) + 1
year_next_scaled = scaler.transform([[year_next]])
prediction_next = model.predict(year_next_scaled)
print(f"\nPrediksi Total Sugar Consumption untuk tahun {year_next}: {prediction_next[0][0]}")

plt.figure(figsize=(8, 6))
plt.scatter(df_avg_per_year['Year'], df_avg_per_year['Total_Sugar_Consumption'], color='blue', label='Data Indonesia')
plt.plot(df_avg_per_year['Year'], model.predict(scaler.transform(df_avg_per_year['Year'].values.reshape(-1, 1))), color='red', label='Prediksi Model')
plt.title('Total Sugar Consumption untuk Indonesia dengan Model Neural Network')
plt.xlabel('Tahun')
plt.ylabel('Total Sugar Consumption')
plt.legend()
plt.show()
