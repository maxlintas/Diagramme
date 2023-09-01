from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


#Pfade zur Excel-Datei und zum Arbeitsblatt (hier wird eine Variable definiert)
excel_file_path = 'C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/electric_heater_clean.xlsx'
sheet_name = 'EK_XY'  # Zu untersuchende Blatt auswählen

#Excel-Datei lesen und Daten in ein DataFrame laden (von Excel zu python tablle)
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# XY-Diagramm erstellen
plt.figure(figsize=(10, 6))  # Einstellen der Größe des Diagramms

# X- und Y-Daten aus dem DataFrame extrahieren
hersteller = df['Hersteller']
max_leistung = df['max. Leistung (MW)']
investitionskosten = df['Spezifische Invest (€/kW) 2023']

# XY-Diagramm erstellen, das hier macht nur punkte
plt.scatter(max_leistung, investitionskosten, label='Hersteller', color='blue')
#show the plot
plt.show()

# Beschriftungen für die Achsen und das Diagramm hinzufügen
plt.xlabel('Maximale Leistung (MW)')
plt.ylabel('Spezifische Investitionskosten (€/kW)')

# Den aktuellen Pfad der Python-Datei abrufen
current_directory = os.path.dirname(os.path.abspath(__file__))

# Alternativ spezifische Spalten in NumPy-Arrays konvertieren
x_data = df['max. Leistung (MW)'].values  # Hier ersetze 'Spaltenname' durch den tatsächlichen Spaltennamen
y_data = df['Spezifische Invest (€/kW) 2023'].values  # Hier ersetze 'Spaltenname' durch den tatsächlichen Spaltennamen

plt.scatter(x_data, y_data, label='Hersteller', color='blue')



from sklearn.linear_model import LinearRegression

# Filtere Datenpunkte, bei denen y_data <= 0 oder x_data <= 0,
# da der Logarithmus von nicht-positiven Werten nicht definiert ist
valid_indices = (y_data > 0) & (x_data > 0)
x_data_valid = x_data[valid_indices]
y_data_valid = y_data[valid_indices]

# Logarithmus der Daten
x_data_log = np.log(x_data_valid)
y_data_log = np.log(y_data_valid)

# Lineare Regression
model = LinearRegression()
model.fit(x_data_log.reshape(-1, 1), y_data_log)

# Koeffizienten extrahieren
b = model.coef_[0]
ln_a = model.intercept_
a = np.exp(ln_a)

print(f"y = {a:.2f} * x^{b:.2f}")

# Vorhersagen treffen und Kurve plotten
x_plot = np.linspace(min(x_data), max(x_data), 100).reshape(-1, 1)
y_plot = a * x_plot**b

plt.scatter(x_data, y_data, label='Datenpunkte', color='blue')
plt.plot(x_plot, y_plot, color='red', label=f'y = {a:.2f} * x^{b:.2f}')
plt.xlabel('Maximale Leistung (MW)')
plt.ylabel('Spezifische Investitionskosten (€/kW)')
plt.legend()
plt.show()


'''Auswertung der Regressionsanalyse'''
from sklearn.metrics import mean_squared_error, r2_score

# Vorhersagen für die Datenpunkte
y_pred = a * x_data_valid**b

# R^2-Wert berechnen
r2 = r2_score(y_data_valid, y_pred)
print(f"R^2-Wert: {r2:.2f}")

# RMSE berechnen
rmse = np.sqrt(mean_squared_error(y_data_valid, y_pred))
print(f"RMSE: {rmse:.2f}")

# Residuenplot
residuen = y_data_valid - y_pred
plt.scatter(x_data_valid, residuen, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Maximale Leistung (MW)')
plt.ylabel('Residuen')
plt.title('Residuenplot')
plt.show()

import statsmodels.api as sm
# Fügen Sie eine Konstante zur unabhängigen Variablen hinzu (für den Intercept)
x_data_log_const = sm.add_constant(x_data_log)

# Erstellen Sie das Modell mit Statsmodels
model_stats = sm.OLS(y_data_log, x_data_log_const)
results = model_stats.fit()

# Rufen Sie eine Zusammenfassung der Regressionsanalyse ab
print(results.summary())

# Rufen Sie den p-Wert für die unabhängige Variable ab (hier die erste Variable)
p_value = results.pvalues[1]  # 1 steht für die erste unabhängige Variable
print(f"p-Wert: {p_value:.4f}")