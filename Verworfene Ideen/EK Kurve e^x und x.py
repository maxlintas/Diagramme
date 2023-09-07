
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

def EK_f(x, a, b, c, d):
    return a * np.exp(-b * x) +c*x+d
#using the curve_fit function
popt, pcov = curve_fit(EK_f, x_data, y_data, p0=[100, 0.5, 28.33, 100])
print('popt=', popt)
print('pcov=', pcov)

#now including the optimal parameters in the plot
a_opt, b_opt, c_opt, d_opt = popt
x_data= np.linspace(min(x_data), max(x_data), 100)
y_data = EK_f(x_data, a_opt, b_opt, c_opt, d_opt)
#plot the data with the curve fit
plt.scatter(x_data, y_data, color='r', linewidths=0.1)
plt.show()
#plot the covariance matrix
plt.imshow(np.log(np.abs(pcov)))
plt.colorbar()
plt.show()