#REGRESI LINEAR MANUAL DENGAN HYPERPARAMETER TUNNING GRID SEARCH DAN OPTIMIZER SGD


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'C:\Users\Fernando\Documents\Telkom_University\S2 Teknik Elektro\Semester 2\Matematika Teknik Lanjut\sugar_consumption_dataset.csv')

print(df.head())

df_filtered = df[df['Country'] == 'Indonesia']
df_filtered_no_duplicates = df_filtered.drop_duplicates(subset='Total_Sugar_Consumption', keep='first')

df_filtered_up_to_2022 = df_filtered_no_duplicates[df_filtered_no_duplicates['Year'] <= 2022]

df_avg_per_year = df_filtered_up_to_2022.groupby('Year')['Total_Sugar_Consumption'].mean().reset_index()

df_avg_per_year.to_csv('avg_per_year_up_to_2022.csv', index=False) 

print(df_avg_per_year.head())

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

print(f"\nRegresi Linear - m: {m_manual}, c: {c_manual}")

year_next = 2023
x_pred_manual = (year_next - np.mean(X)) / np.std(X)
prediction_2023_manual = m_manual * x_pred_manual + c_manual
print(f"Prediksi Total Sugar Consumption untuk tahun {year_next} (manual): {prediction_2023_manual}")

df_filtered_2023 = df_filtered[df_filtered['Year'] == 2023]
average_2023 = df_filtered_2023['Total_Sugar_Consumption'].mean() if not df_filtered_2023.empty else None

if average_2023 is not None:
    print(f"Rata-rata Total Sugar Consumption untuk tahun 2023: {average_2023}")
    print(f"\nPerbandingan antara rata-rata tahun 2023 dan prediksi:")
    print(f"Rata-rata Total Sugar Consumption 2023: {average_2023}")
    print(f"Prediksi Total Sugar Consumption 2023 (manual): {prediction_2023_manual}")
else:
    print(f"Tidak ada data untuk tahun 2023, hanya menggunakan prediksi.")

# SGD Manual
alpha_values = [0.00001, 0.00005, 0.0001]
batch_size_values = [5, 10, 20]
num_iter_values = [1000, 2000, 3000]

best_mse = float('inf')
best_params = None
best_m = 0
best_c = 0

for alpha in alpha_values:
    for batch_size in batch_size_values:
        for num_iter in num_iter_values:
            m = 0
            c = 0
            n = len(X)

            objective_val = []

            for i in range(num_iter):
                idx = np.random.choice(n, batch_size)
                xi_batch = X_normalized[idx]
                yi_batch = y[idx]

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
                    print("MSE menjadi NaN, menghentikan iterasi.")
                    break

            if mse < best_mse:
                best_mse = mse
                best_params = (alpha, batch_size, num_iter)
                best_m = m
                best_c = c

print("\nNilai Optimal m (SGD):", best_m)
print("Nilai Optimal c (SGD):", best_c)
print(f"Hyperparameters terbaik - Alpha: {best_params[0]}, Batch Size: {best_params[1]}, Num Iterations: {best_params[2]}")

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
plt.title("Konvergensi Gradient Descent (SGD)")

plt.tight_layout()
plt.show()
