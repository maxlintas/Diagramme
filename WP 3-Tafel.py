import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import SmoothBivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime#um datum in dateinamen aufzunehmen
import seaborn as sns# visualisierung mit heatmap
from scipy.stats import pearsonr# um korrelation zu berechnen

# Pfade zur Excel-Datei und zum Arbeitsblatt
excel_file_path = 'C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/electric_heater_clean.xlsx'
sheet_name = 'WP_XY'
# Excel-Datei lesen und Daten in ein DataFrame laden
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
#check if invalid data is in collumns that im using for diagramm:
columns_to_check = ['max. Leistung', 'Spezifische Invest (€/kW) 2023', 'Senke-Aus']  # Fügen Sie hier die Spaltennamen hinzu, die Sie überprüfen möchten
print('es gibt fehlerhafte Zeilen:',df[columns_to_check].isna().sum())

# Farbpalette für die Hersteller generieren
unique_manufacturers = df['Hersteller'].unique()
color_palette = cm.tab20(np.linspace(0, 1, 11)) # len(unique_manufacturers)) um gleichvieel anzahl an farben wie hersteller zu haben, leider habe ich aber mehr hersteller als ich gerade betrachte, deshlab manuell 11

#daten aus dataframe extrahieren
T_max = df['Senke-Aus']
investitionskosten = df['Spezifische Invest (€/kW) 2023']
# Konvertiere die 'max. Leistung'-Spalte in numerischen Datentyp
max_leistung = pd.to_numeric(df['max. Leistung'], errors='coerce')
hersteller = df['Hersteller']

# Erstellen Sie eine 2x2-Gitterstruktur für die Diagramme
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
#axs[0,0]=oben links, 0,1=oben rechts, 1,0=unten links, 1,1=unten rechts

# XY-Diagramm (unten rechts)
for idx, manufacturer in enumerate(unique_manufacturers):
    manufacturer_data = df[df['Hersteller'] == manufacturer]
    x = manufacturer_data['max. Leistung']
    y = manufacturer_data['Spezifische Invest (€/kW) 2023']
    axs[1, 1].scatter(x, y, label=manufacturer, color=color_palette[idx % 11])
axs[1, 1].set_xlabel('max. Leistung [MW]')
axs[1, 1].set_ylabel('Spezifische Investitionskosten [€/kW]')

# XZ-Diagramm (unten links)
for idx, manufacturer in enumerate(unique_manufacturers):
    manufacturer_data = df[df['Hersteller'] == manufacturer]
    x = manufacturer_data['max. Leistung']
    z = manufacturer_data['Senke-Aus']
    axs[1, 0].scatter(x, z, label=manufacturer, color=color_palette[idx % 11])
axs[1, 0].set_xlabel('max. Leistung [MW]')
axs[1, 0].set_ylabel('Senke-Aus [°C]')

# YZ-Diagramm (oben rechts)
for idx, manufacturer in enumerate(unique_manufacturers):
    manufacturer_data = df[df['Hersteller'] == manufacturer]
    y = manufacturer_data['Spezifische Invest (€/kW) 2023']
    z = manufacturer_data['max. Leistung']
    axs[0, 1].scatter(z, y, label=manufacturer, color=color_palette[idx % 11])
axs[0, 1].set_xlabel('Leistung [MW]')
axs[0, 1].set_ylabel('Spezifische Investitionskosten [€/kW]')

# Das obere linke Diagramm wird nicht verwendet, deshalb legende hier rein
handles, labels = axs[1, 1].get_legend_handles_labels() #legt legende fest
axs[0,0].legend(handles, labels, loc='center') #legt legende fest
axs[0, 0].axis('off')#blendet achsen aus

# Diagramm anzeigen
plt.tight_layout()
plt.show()

# Aktuelles Datum generieren
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Dateinamen mit aktuellem Datum formatieren
file_name = 'C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/Bilder/WP - charts/wp 3-Tafel {}.png'.format(current_date)

# Diagramm speichern
fig.savefig(file_name, dpi=300)

'''Korrelationen untersuchen und ausgeben'''
#spalten auswählen die ich für korrelationen untersuchen will
selected_columns = df[['Senke-Aus', 'Spezifische Invest (€/kW) 2023', 'max. Leistung']]
#Korrelationsmatrix aus meinen df machen
correlation_matrix = selected_columns.corr()
#heatmap ausgeben um visualisierung der korrelationen zu haben
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Temperatur und Investitionskosten untersuchen
x = df['Senke-Aus']
y = df['Spezifische Invest (€/kW) 2023']

# Untersuchung ob Korrelation signifikant ist: x und y sind die beiden zu vergleichenden Variablen
corr, p_value = pearsonr(x,y)
if p_value < 0.05:
    print("Korrelation zwischen {} und {} ist statistisch signifikant mit einem p-Wert von {}.".format(x.name, y.name, p_value))
else:
    print("Keine statistisch signifikante Korrelation zwischen {} und {}.".format(x.name, y.name))

#Leistung und Investitionskosten untersuchen
x = df['max. Leistung']
y = df['Spezifische Invest (€/kW) 2023']
# Untersuchung ob Korrelation signifikant ist: x und y sind die beiden zu vergleichenden Variablen
corr, p_value = pearsonr(x,y)
if p_value < 0.05:
    print("Korrelation zwischen {} und {} ist statistisch signifikantmit einem p-Wert von {}.".format(x.name, y.name, p_value))
else:
    print("Keine statistisch signifikante Korrelation zwischen {} und {}.".format(x.name, y.name))