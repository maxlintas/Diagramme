import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Pfade zur Excel-Datei und zum Arbeitsblatt (hier wird eine Variable definiert)
excel_file_path = 'C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/electric_heater_clean.xlsx'
sheet_name = 'WP_XY'  # Zu untersuchende Blatt auswählen

# Excel-Datei lesen und Daten in ein DataFrame laden (von Excel zu python tablle)
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

'''Nicht-lineare Regression mit Curve_fit()'''
# Definieren der Funktion basierend auf der gegebenen Gleichung
def cop_eq(data, a, b, c, d):
    T_lift, T_sa = data
    return (a * (T_lift + 2 * b) ** c) * ((T_sa + b) ** d)

T_lift = df['T_lift,m [K]']
T_sa = df['T_h,m [K]']
COP = df['COP']
#check if there are NaN values, that could cause problems
print("NaN in T_lift:", T_lift.isna().sum())
print("NaN in T_sa:", T_sa.isna().sum())
print("NaN in COP:", COP.isna().sum())

# Fit der Daten an die Gleichung mit curve_fit
initial_guess = [100, 1, -1, 0]  # Initiale Schätzwerte für die Parameter
lower_bounds = [0, 0, -np.inf, -np.inf]
upper_bounds = [np.inf, np.inf, np.inf, np.inf]
fit_params, _ = curve_fit(cop_eq, (T_lift,T_sa), COP, initial_guess, maxfev=100000)

# Extrahieren der Parameter
a_fit, b_fit, c_fit, d_fit = fit_params

# Funktionsterm erstellen
function_term = f'COP = {a_fit:.2f} * (T_lift + 2 * {b_fit:.2f}) ** {c_fit:.2f} * (T_sa + {b_fit:.2f}) ** {d_fit:.2f}'
print(f'Funktionsterm der Gleichung: {function_term}')
print (f'Funktionsparameter der Gleichung: {fit_params}')

#statistische Kennzahlen

residuals = COP - cop_eq((T_lift, T_sa), a_fit, b_fit, c_fit, d_fit)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((COP - np.mean(COP))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f'R²-Wert: {r_squared:.2f}, Achtung, bei nicht-linearen Modellen ist der R²-Wert nicht aussagekräftig!')

mse = np.mean(residuals**2)
X = np.column_stack((T_lift, T_sa))
std_err = np.sqrt(np.diag(mse * np.linalg.inv(np.dot(X.T, X))))
print(f'Standardfehler der Regression: {std_err}, Achtung, bei nicht-linearen Modellen ist der Standardfehler der Regression nicht aussagekräftig!')

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Berechnung der vorhergesagten Werte mit dem gefitteten Modell
predicted_COP = cop_eq((T_lift, T_sa), a_fit, b_fit, c_fit, d_fit)

# RMSE berechnen
rmse = np.sqrt(mean_squared_error(COP, predicted_COP))
print(f'Root Mean Square Error (RMSE) for non-linear regression: {rmse:.2f}')

# MAE berechnen
mae = mean_absolute_error(COP, predicted_COP)
print(f'Mean Absolute Error (MAE) for non linear-regression: {mae:.2f}')

print('Je näher die Werte bei 0 liegen, desto besser passt das Modell. Jetzt können Sie die Werte mit anderen Modellen vergleichen.')

# Plotten der Oberfläche
T_lift_values = np.linspace(min(T_lift), max(T_lift), 100)
T_sa_values = np.linspace(min(T_sa), max(T_sa), 100)
T_lift_grid, T_sa_grid = np.meshgrid(T_lift_values, T_sa_values)
cop_predicted = cop_eq((T_lift_grid, T_sa_grid), a_fit, b_fit, c_fit, d_fit)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(T_lift_grid, T_sa_grid, cop_predicted, cmap='viridis', alpha=0.5)

ax.scatter(T_lift, T_sa, COP, color='blue', label='Datenpunkte')

ax.set_xlabel('T_lift [°C]')
ax.set_ylabel('T_sa [°C]')
ax.set_zlabel('COP')

plt.show()

# Durchschnittswerte berechnen
avg_T_sa = np.mean(T_sa)
avg_T_lift = np.mean(T_lift)

# Sortieren der Daten für T_lift
sorted_indices = np.argsort(T_lift)
T_lift_sorted = T_lift[sorted_indices]
COP_sorted_by_T_lift = COP[sorted_indices]

# Plotting COP against T_lift with the calculated equation
plt.figure(figsize=(10, 6))
plt.scatter(T_lift_sorted, COP_sorted_by_T_lift, label='Actual Data', color='blue')
plt.plot(T_lift_sorted, cop_eq((T_lift_sorted, np.full_like(T_lift_sorted, avg_T_sa)), a_fit, b_fit, c_fit, d_fit), label='Fitted Equation', color='red')
plt.xlabel('T_lift [°C]')
plt.ylabel('COP')
plt.title('COP vs T_lift')
plt.legend()
plt.show()

# Sortieren der Daten für T_sa
sorted_indices_sa = np.argsort(T_sa)
T_sa_sorted = T_sa[sorted_indices_sa]
COP_sorted_by_T_sa = COP[sorted_indices_sa]

# Plotting COP against T_sa with the calculated equation
plt.figure(figsize=(10, 6))
plt.scatter(T_sa_sorted, COP_sorted_by_T_sa, label='Actual Data', color='blue')
plt.plot(T_sa_sorted, cop_eq((np.full_like(T_sa_sorted, avg_T_lift), T_sa_sorted), a_fit, b_fit, c_fit, d_fit), label='Fitted Equation', color='red')
plt.xlabel('T_sa [°C]')
plt.ylabel('COP')
plt.title('COP vs T_sa')
plt.legend()
plt.show()



'''Multivariante lineare Regression als Alternative zur nicht-linearen Regression'''
print('''Multivariante lineare Regression als Alternative zur nicht-linearen Regression''')
# Erstellen Sie die Designmatrix X
X = pd.DataFrame({'T_sa': T_sa, 'T_lift': T_lift})
X = sm.add_constant(X)  # Fügt eine Konstante (Intercept) zur Designmatrix hinzu

# Erstellen Sie das Modell
model = sm.OLS(COP, X)

# Führen Sie die Regression durch
results = model.fit()

# Zugriff auf die Koeffizienten (Parameter)
print("Koeffizienten: ", results.params)

# Der Funktionsterm
print(f"Funktionsterm: y = {results.params[0]} + {results.params[1]}*T_sa + {results.params[2]}*T_lift")
#Funktionsterm mit beiden unabhaengigen variablen:
def predict_cop(T_sa, T_lift):
    return 4.487 + 0.000214 * T_sa - 0.0208 * T_lift
print('Funktionsterm mit beiden unabhaengigen variablen: predict_cop(T_sa, T_lift)= 4.487 + 0.000214 * T_sa - 0.0208 * T_lift')

predicted_COP_linear = results.predict(X)

# RMSE und MAE für die lineare Regression
rmse_linear = np.sqrt(mean_squared_error(COP, predicted_COP_linear))
mae_linear = mean_absolute_error(COP, predicted_COP_linear)
print(f'Root Mean Square Error (RMSE) für lineare Regression: {rmse_linear:.2f}')
print(f'Mean Absolute Error (MAE) für lineare Regression: {mae_linear:.2f}')

# Ausgabe der Ergebnisse
print('Ergebnisse der linearen Regression:')
print(results.summary())

# Angenommen, COP und y_pred sind Ihre Pandas-Serien mit den tatsächlichen und vorhergesagten Werten
COP = pd.Series([COP])
y_pred = pd.Series([predicted_COP])



# Check shapes
print("Shape of y_true:", y_true.shape)
print("Shape of y_pred:", y_pred.shape)

# Reshape y_pred if necessary
y_pred = y_pred.reshape(-1)

# Calculate Mean Squared Error (MSE)
if y_true.shape != y_pred.shape:
    print("Shapes do not match!")
else:
    mse = mean_squared_error(y_true, y_pred)
    print(f"Mean Squared Error: {mse}")

