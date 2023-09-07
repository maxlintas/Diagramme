
import pandas as pd
import matplotlib.pyplot as plt
import os

# Pfade zur Excel-Datei und zum Arbeitsblatt
excel_file_path = 'C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/electric_heater_clean.xlsx'
sheet_name = 'WP_XY'

# Excel-Datei lesen und Daten in ein DataFrame laden
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Daten im DataFrame anzeigen (optional)
print(df)

# X- und Y-Daten aus dem DataFrame extrahieren
hersteller = df['Hersteller']
max_leistung = df['max. Leistung']
investitionskosten = df['Spezifische Invest (€/kW) 2023']

# XY-Diagramm erstellen
plt.figure(figsize=(10, 6))  # Einstellen der Größe des Diagramms

# XY-Diagramm erstellen
plt.scatter(max_leistung, investitionskosten, label='Hersteller', color='blue')

# Beschriftungen für die Achsen und das Diagramm hinzufügen
plt.xlabel('Maximale Leistung (MW)')
plt.ylabel('Spezifische Investitionskosten (€/kW)')
plt.title('Wärmepumpen-Hersteller - Spezifische Investitionskosten zu maximaler Leistung -')

# Diagramm anzeigen
plt.show()

# Den aktuellen Pfad der Python-Datei abrufen
current_directory = os.path.dirname(os.path.abspath(__file__))

# DataFrame mit den gefilterten Zeilen in Excel-Datei speichern
output_excel_file_path = os.path.join(current_directory, 'WP XY spez.invest zu max.leistung.xlsx')
df.to_excel(output_excel_file_path, index=False)

#plot als png speichern
plt.savefig('C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/Bilder/WP-charts/WP XY spez.invest zu max.leistung.png')