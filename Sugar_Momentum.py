import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'file.csv')

df = df.dropna()

df_filtered = df[df['Country'] == 'Indonesia']
df_filtered_no_duplicates = df_filtered.drop_duplicates(subset='Total_Sugar_Consumption', keep='first')

df_avg_per_year = df_filtered_no_duplicates.groupby('Year')['Total_Sugar_Consumption'].mean().reset_index()

X = df_avg_per_year['Year'].values.reshape(-1, 1)
y = df_avg_per_year['Total_Sugar_Consumption'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))

model.compile(optimizer=SGD(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1)

y_pred = model.predict(X_test)

print(f"y_test contains NaN: {np.any(np.isnan(y_test))}")
print(f"y_pred contains NaN: {np.any(np.isnan(y_pred))}")

y_pred = np.nan_to_num(y_pred, nan=0.0)

mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error pada data uji: {mse}")

year_2023 = 2023
year_2023_scaled = scaler.transform([[year_2023]])
prediction_2023 = model.predict(year_2023_scaled)
print(f"\nPrediksi Total Sugar Consumption untuk tahun 2023: {prediction_2023[0][0]}")

real_value_2023 = df_filtered_no_duplicates[df_filtered_no_duplicates['Year'] == 2023]
if not real_value_2023.empty:
    print(f"Total Sugar Consumption yang sebenarnya pada tahun 2023: {real_value_2023['Total_Sugar_Consumption'].values[0]}")
else:
    print("Tidak ada data untuk tahun 2023 dalam dataset.")

plt.figure(figsize=(8, 6))
plt.scatter(df_avg_per_year['Year'], df_avg_per_year['Total_Sugar_Consumption'], color='blue', label='Data Indonesia')
plt.plot(df_avg_per_year['Year'], model.predict(scaler.transform(df_avg_per_year['Year'].values.reshape(-1, 1))), color='red', label='Prediksi Model')
plt.title('Total Sugar Consumption untuk Indonesia dengan SGD Optimizer')
plt.xlabel('Tahun')
plt.ylabel('Total Sugar Consumption')
plt.legend()
plt.show()
