
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
#print(df)
# XY-Diagramm erstellen
plt.figure(figsize=(10, 6))  # Einstellen der Größe des Diagramms

# Filtere Zeilen mit "Ja" in Spalte A
#df_filtered = df[df['verwendbar'] == 'ja']

# X- und Y-Daten aus dem DataFrame extrahieren
hersteller = df['Hersteller']
max_leistung = df['max. Leistung (MW)']
investitionskosten = df['Spezifische Invest (€/kW) 2023']

# XY-Diagramm erstellen, das hier macht nur punkte
plt.scatter(max_leistung, investitionskosten, label='Hersteller', color='blue')

# Beschriftungen für die Achsen und das Diagramm hinzufügen
plt.xlabel('Maximale Leistung (MW)')
plt.ylabel('Spezifische Investitionskosten (€/kW)')
#plt.title('EK-Hersteller - Spezifische Invest - max Leistung -')

# Diagramm anzeigen
#plt.show()

# Den aktuellen Pfad der Python-Datei abrufen
current_directory = os.path.dirname(os.path.abspath(__file__))

# DataFrame mit den gefilterten Zeilen in Excel-Datei speichern
#output_excel_file_path = os.path.join(current_directory, 'EK Hersteller spezifische Investitionskosten.xlsx')
#df_filtered.to_excel(output_excel_file_path, index=False)

#plot als png speichern
#plt.savefig('C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/Bilder/EK-charts/EK spzezInvest, max. Leistung.png', dpi=300)

#DataFrame mit den gefilterten Zeilen in Excel-Datei speichern
#output_excel_file_path = 'pfad_zur_gespeicherten_datei.xlsx'  # Ersetze 'pfad_zur_gespeicherten_datei.xlsx' durch den gewünschten Speicherpfad
#df_filtered.to_excel(output_excel_file_path, index=False)

#df.to_csv('C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/Bilder/EK-charts/EK spzezInvest, max. Leistung.csv', index=False)

# Daten aus dem DataFrame in NumPy-Arrays konvertieren
#array_from_dataframe = df.values  # Alle Spalten werden in ein NumPy-Array umgewandelt

# Alternativ kannst du spezifische Spalten in NumPy-Arrays konvertieren
x_data = df['max. Leistung (MW)'].values  # Hier ersetze 'Spaltenname' durch den tatsächlichen Spaltennamen
y_data = df['Spezifische Invest (€/kW) 2023'].values  # Hier ersetze 'Spaltenname' durch den tatsächlichen Spaltennamen

plt.scatter(x_data, y_data, label='Hersteller', color='blue')

#plt.show()

# Alternativ kannst du spezifische Zeilen in NumPy-datas konvertieren
def EK_f(x, a, b, c):
    return a*(x-b)**2+c
#using the curve_fit function
popt, pcov = curve_fit(EK_f, x_data, y_data, p0=[3, 60, 60])
print('popt=', popt)
print('pcov=', pcov)

#now including the optimal parameters in the plot
a_opt, b_opt, c_opt = popt
x_data= np.linspace(min(x_data), max(x_data), 100)
y_data = EK_f(x_data, a_opt, b_opt, c_opt)
#plot the data with the curve fit
plt.scatter(x_data, y_data, color='r', linewidths=0.1)
plt.show()
#plot the covariance matrix
plt.imshow(np.log(np.abs(pcov)))
plt.colorbar()
plt.show()