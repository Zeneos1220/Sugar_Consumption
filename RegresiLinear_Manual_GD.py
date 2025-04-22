#REGRESI LINEAR MANUAL DENGAN HYPERPARAMETER TUNNING GRID SEARCH DAN OPTIMIZER GD

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'filenya.csv')

df_filtered = df[df['Country'] == 'Indonesia']
df_filtered_no_duplicates = df_filtered.drop_duplicates(subset='Total_Sugar_Consumption', keep='first')
df_filtered_up_to_2022 = df_filtered_no_duplicates[df_filtered_no_duplicates['Year'] <= 2022]
df_avg_per_year = df_filtered_up_to_2022.groupby('Year')['Total_Sugar_Consumption'].mean().reset_index()
df_avg_per_year.to_csv('avg_per_year_up_to_2022.csv', index=False)

X = df_avg_per_year['Year'].values.reshape(-1, 1)
y = df_avg_per_year['Total_Sugar_Consumption'].values
X_normalized = (X - np.mean(X)) / np.std(X)

x_manual = X_normalized.flatten()
y_manual = y
n_manual = len(x_manual)

sum_x = np.sum(x_manual)
sum_y = np.sum(y_manual)
sum_xy = np.sum(x_manual * y_manual)
sum_x2 = np.sum(x_manual ** 2)

m_manual = (n_manual * sum_xy - sum_x * sum_y) / (n_manual * sum_x2 - sum_x**2)
c_manual = (sum_y - m_manual * sum_x) / n_manual

year_next = 2023
x_pred_manual = (year_next - np.mean(X)) / np.std(X)
prediction_2023_manual = m_manual * x_pred_manual + c_manual

df_filtered_2023 = df_filtered[df_filtered['Year'] == 2023]
average_2023 = df_filtered_2023['Total_Sugar_Consumption'].mean() if not df_filtered_2023.empty else None

if average_2023 is not None:
    print(f"Rata-rata Total Sugar Consumption 2023: {average_2023}")
    print(f"Prediksi Total Sugar Consumption 2023 (manual): {prediction_2023_manual}")
else:
    print(f"Tidak ada data untuk tahun 2023, hanya menggunakan prediksi.")
    print(f"Prediksi Total Sugar Consumption 2023 (manual): {prediction_2023_manual}")

alpha_values = [0.00001, 0.00005, 0.0001]
num_iter_values = [1000, 2000, 3000]

best_mse = float('inf')
best_params = None
best_m = 0
best_c = 0

for alpha in alpha_values:
    for num_iter in num_iter_values:
        m = 0
        c = 0
        objective_val = []

        for i in range(num_iter):
            xi_batch = X_normalized
            yi_batch = y

            y_pred_batch = m * xi_batch + c
            error = yi_batch - y_pred_batch

            dm = -2 * np.mean(xi_batch * error)
            dc = -2 * np.mean(error)

            dm = np.clip(dm, -10, 10)
            dc = np.clip(dc, -10, 10)

            m -= alpha * dm
            c -= alpha * dc

            mse = np.mean((y - (m * X_normalized.flatten() + c)) ** 2)
            objective_val.append(mse)

            if np.isnan(mse):
                break

        if mse < best_mse:
            best_mse = mse
            best_params = (alpha, num_iter)
            best_m = m
            best_c = c

print("\nNilai Optimal m (GD):", best_m)
print("Nilai Optimal c (GD):", best_c)
print(f"Hyperparameters terbaik - Alpha: {best_params[0]}, Num Iterations: {best_params[1]}")

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label="Data Indonesia")
y_pred_manual_line = m_manual * X_normalized.flatten() + c_manual
plt.plot(X, y_pred_manual_line, color='red', label='Regresi Linear (Manual)')
plt.xlabel("Tahun")
plt.ylabel("Total Sugar Consumption")
plt.legend()
plt.title("Regresi Linier (Manual)")

plt.subplot(1, 2, 2)
plt.plot(range(len(moving_average(objective_val))), moving_average(objective_val), color="green")
plt.xlabel("Iterasi")
plt.ylabel("Nilai Fungsi Objektif (MSE)")
plt.title("Konvergensi Gradient Descent (GD)")

plt.tight_layout()
plt.show()
