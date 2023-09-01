
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

#Pfade zur Excel-Datei und zum Arbeitsblatt (hier wird eine Variable definiert)
excel_file_path = 'C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/electric_heater_clean.xlsx'
sheet_name = 'EK_XY'  # Zu untersuchende Blatt auswählen

#Excel-Datei lesen und Daten in ein DataFrame laden (von Excel zu python tablle); df ist eine variable
df = pd.read_excel(excel_file_path, sheet_name=sheet_name) #pd.read ist eine funktion von pandas; sheet_name ist eine variable die als Argument in die Funktion gegeben wird

#df anzeigen lassen
print(df)
# XY-Diagramm erstellen
plt.figure(figsize=(10, 6))  # Einstellen der Größe des Diagramms

# X- und Y-Daten aus dem DataFrame extrahieren
hersteller = df['Hersteller']
max_leistung = df['max. Leistung (MW)']
investitionskosten = df['Spezifische Invest (€/kW) 2023']

# XY-Diagramm erstellen (das hier macht nur punkte)
plt.scatter(max_leistung, investitionskosten, label='Hersteller', color='blue')

# Beschriftungen für die Achsen und das Diagramm hinzufügen
plt.xlabel('Maximale Leistung (MW)')
plt.ylabel('Spezifische Investitionskosten (€/kW)')

# Diagramm anzeigen
plt.show()

# Den aktuellen Pfad der Python-Datei abrufen
current_directory = os.path.dirname(os.path.abspath(__file__))

# Alternativ kannst du spezifische Spalten in NumPy-Arrays konvertieren
x_data = df['max. Leistung (MW)'].values  # Hier ersetze 'Spaltenname' durch den tatsächlichen Spaltennamen
y_data = df['Spezifische Invest (€/kW) 2023'].values  # Hier ersetze 'Spaltenname' durch den tatsächlichen Spaltennamen
plt.scatter(x_data, y_data, label='Hersteller', color='blue')

y_data = df['Spezifische Invest (€/kW) 2023'].values  # Hier ersetze 'Spaltenname' durch den tatsächlichen Spaltennamen


# spezifische Zeilen in NumPy-datas konvertieren
def EK_f(x, a, b,c):
    return a * np.exp(-b * x) +c #ggf noch term hinzufügen, der als exponentialfunktion die elektrodenkessel beschreibt
#using the curve_fit function
popt, pcov = curve_fit(EK_f, x_data, y_data, p0= [1000, 0.5, 15.33])
print('popt=', popt)
print('pcov=', pcov)

#now including the optimal parameters in the plot
a_opt, b_opt, c_opt = popt
x_data = np.linspace(min(x_data), max(x_data), 100)
y_data = EK_f(x_data, a_opt, b_opt, c_opt)
#plot the data with the curve fit
plt.scatter(x_data, y_data, color='r', linewidths=0.1)
plt.show()
#plot the covariance matrix
plt.imshow(np.log(np.abs(pcov)))
plt.colorbar()
plt.show()

'''Confidence and prediction band'''
# Residuen berechnen
residuals = y_data - EK_f(x_data, *popt)
# Standardabweichung der Residuen
std_dev = np.std(residuals)

# Prediction-Band berechnen
t = np.linspace(min(x_data), max(x_data), 100)
y_pred = EK_f(t, *popt)
y_lower = y_pred - 1.96 * std_dev  # 95% Konfidenzniveau
y_upper = y_pred + 1.96 * std_dev  # 95% Konfidenzniveau

# Daten, Kurvenanpassung und Prediction-Band plotten
plt.scatter(x_data, y_data, label='Hersteller', color='blue')
plt.plot(t, y_pred, 'r-', label='Fit')
plt.fill_between(t, y_lower, y_upper, color='red', alpha=0.2, label='95% Prediction Band')
plt.legend()
plt.show()


'''Kolmogorov-Smirnov-Test'''
# Residuen berechnen
residuals = y_data - EK_f(x_data, *popt)

# Kolmogorov-Smirnov-Test
from scipy.stats import kstest
ks_statistic, p_value = kstest(residuals, 'norm')
print('KS-Statistik:', ks_statistic)
print('p-Wert:', p_value)

'''standard error of the regression'''
# Compute the standard errors
perr = np.sqrt(np.diag(pcov))
for i, param in enumerate(popt):
    print(f"Parameter {i}: Value = {param}, Standard Error = {perr[i]}")
# Predicted values from the model
y_pred = EK_f(x_data, *popt)

plt.errorbar(x_data, y_pred, yerr=perr, fmt='o', label='Fitted values with SE')
plt.plot(x_data, y_data, 'r.', label='Original data')
plt.legend()
plt.show()