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
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Pfade zur Excel-Datei und zum Arbeitsblatt
excel_file_path = 'C:/Users/m.wirth/OneDrive - Universitaet Duisburg-Essen/1 Masterarbeit/Technologien/electric_heater_clean.xlsx'
sheet_name = 'WP_XY'
# Excel-Datei lesen und Daten in ein DataFrame laden
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

#check for empty data in collumns that im using for diagramm:
columns_to_check = ['max. Leistung', 'Spezifische Invest (€/kW) 2023', 'Senke-Aus']
# Überprüfen Sie, ob es fehlende Werte in den Spalten gibt
missing_values = df[columns_to_check].isna().sum()
# Überprüfen Sie, ob es nicht-numerische Werte in den Spalten gibt
non_numeric_values = df[columns_to_check].apply(lambda col: pd.to_numeric(col, errors='coerce')).isna().sum()
# Wenn es fehlende oder nicht-numerische Werte in einer der Spalten gibt, geben Sie eine Nachricht aus
if missing_values.sum() > 0 or non_numeric_values.sum() > 0:
    print('Es gibt fehlerhafte Zeilen:')
    print('Fehlende Werte:', missing_values)
    print('Nicht-numerische Werte:', non_numeric_values)


# Farbpalette für die Hersteller generieren
unique_manufacturers = df['Hersteller'].unique()
color_palette = cm.tab20(np.linspace(0, 1, 11)) # len(unique_manufacturers)) um gleichvieel anzahl an farben wie hersteller zu haben, leider habe ich aber mehr hersteller als ich gerade betrachte, deshlab manuell 11

#daten aus dataframe extrahieren
T_max = df['Senke-Aus']
investitionskosten = df['Spezifische Invest (€/kW) 2023']
# Konvertiere die 'max. Leistung'-Spalte in numerischen Datentyp
max_leistung = pd.to_numeric(df['max. Leistung'], errors='coerce')
hersteller = df['Hersteller']

#3D Diagramm erstellen
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

# Schleife über die einzigartigen Herstellernamen
for idx, manufacturer in enumerate(unique_manufacturers):
    manufacturer_data = df[df['Hersteller'] == manufacturer]
    x = manufacturer_data['max. Leistung']
    y = manufacturer_data['Spezifische Invest (€/kW) 2023']
    z = manufacturer_data['Senke-Aus']
    #ax.scatter(x,y,z, label=manufacturer, color=color_palette[idx % 11])
    # Datenpunkte auf dem Boden des Diagramms zeichnen
    ax.scatter(x, y, np.zeros_like(z), label=manufacturer, color=color_palette[idx % 11])
    # Datenpunkte an der linken Wand des Diagramms zeichnen
    ax.scatter(np.zeros_like(x), y, z, label=manufacturer, color=color_palette[idx % 11])


# Benutzerdefinierte xyz-Achsenticks festlegen
ax.set_xticks(range(0, 51, 5))  # Hier werden Ticks bei 0, 10, 20, ..., 100 gesetzt
ax.set_yticks(range(200, 1401, 100))  # Hier werden Ticks bei 0, 10, 20, ..., 100 gesetzt
ax.set_zticks(range(0, 201, 20))  # Hier werden Ticks bei 0, 10, 20, ..., 100 gesetzt

ax.set_xlabel('max. Leistung [MW]')
ax.set_ylabel('Spezifische Investitionskosten [€/kW]')
ax.set_zlabel('Senke-Aus [°C]')

# Diagramm anzeigen
plt.show()

# Den aktuellen Pfad der Python-Datei abrufen
current_directory = os.path.dirname(os.path.abspath(__file__))

# # Funktion zur Berechnung der Kurve
# def curve_function(x, y, a, b, c):
#     return a * x + b * y + c
#
# # Funktion zur Berechnung der Kosten (curve_fit)


'''Spline fitting'''
#mit spline fitting:
# x1 = np.array(max_leistung)
# x2 = np.array(T_max)
# y = np.array(investitionskosten)
#
#
# #spline fitting mit scipy funktioniert nicht, da die daten nicht gleichverteilt sind??
# #spline fitting
# spline = SmoothBivariateSpline(x, y, z, kx=2, ky=2)
#
# #Neue Werte für Vorhersage
# new_z = np.linspace(min(z), max(z), 100)
# new_x = np.linspace(min(x), max(x), 100)
#
# # Meshgrid erstellen
# x, z = np.meshgrid(new_z, new_x)
#
# # Geschätzte Investitionskosten für das Meshgrid
# predicted_invest_costs = spline(x, z)
#
# # 3D-Diagramm anzeigen
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, z, predicted_invest_costs, cmap='viridis')
#
# ax.set_xlabel('Temperatur')
# ax.set_ylabel('Maximale Leistung')
# ax.set_zlabel('Investitionskosten')
# plt.show()

'''Polynomial fitting'''
x1 = T_max.values.reshape(-1, 1)
x2 = max_leistung.values.reshape(-1, 1)
y = investitionskosten.values

# Choose the degrees of the polynomials
degree_x1 = 1
degree_x2 = 1

# Transform the data to include polynomial features
poly_features = PolynomialFeatures(degree=[degree_x1, degree_x2])
x_poly = poly_features.fit_transform(np.hstack((x1, x2)))

# Fit a linear regression model to the polynomial features
model = LinearRegression()
model.fit(x_poly, y)

'''Funktionsterm ausgeben'''
# Die Koeffizienten des Polynommodells abrufen
coefficients = model.coef_
intercept = model.intercept_

# Erstelle die Namen der Polynomfeatures manuell
poly_feature_names = []
for i in range(degree_x1 + 1):
    for j in range(degree_x2 + 1):
        if i + j > 0:
            term = f'x1^{i} * x2^{j}'
            poly_feature_names.append(term)

# Den Funktionsterm des Polynommodells erstellen
function_term = f'{intercept:.2f} + '
function_term += ' + '.join([f'{coef:.2f} * {poly_feature_names[i]}' for i, coef in enumerate(coefficients)])

print(f'Funktionsterm des Polynommodells: Spez. Investk. = {function_term}')
print('Mit x1 = Senken-Austritts-Temperatur und x2 = max. Leistung')


'''Plotting the surface'''

# Generating a meshgrid of values for plotting
x1_values = np.linspace(min(x1), max(x1), 100)
x2_values = np.linspace(min(x2), max(x2), 100)
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

# Transform the meshgrid to polynomial features
x_grid_poly = poly_features.transform(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))

# Predict investment costs using the model
predicted_invest_costs = model.predict(x_grid_poly)

# Reshape the predictions back to the grid shape
predicted_invest_costs_grid = predicted_invest_costs.reshape(x1_grid.shape)

# Plotting the surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x1_grid, x2_grid, predicted_invest_costs_grid, cmap='viridis', alpha=0.5)

ax.scatter(x1, x2, y, color='blue', label='Datenpunkte')

ax.set_xlabel('Senke-Aus [°C]')
ax.set_ylabel('Maximale Leistung [MW]')
ax.set_zlabel('Spezifische Investitionskosten [€/kW]')

plt.show()

def func(x1, x2):
    return 377.35 + 0.00 * x1**0 * x2**1 + 0.00 * x1**0 * x2**2 + 0.01 * x1**0 * x2**3 + -0.30 * x1**1 * x2**0 + 2.72 * x1**1 * x2**1

# Erzeugen von x1- und x2-Werten
x1_vals = np.linspace(min(x1), max(x1), 100)
x2_vals = np.linspace(min(x2), max(x2), 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
z = func(x1_grid, x2_grid)

# Erstellen des 2D-Plots
plt.imshow(z, extent=(x1_vals.min(), x1_vals.max(), x2_vals.min(), x2_vals.max()), origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='f(x1, x2)')
plt.xlabel('Senken-Temperature [°C]')
plt.ylabel('Leistung [MW]')
plt.show()

# Vorhersagen mit dem Modell
y_pred = model.predict(x_poly)

# R-Quadrat-Wert berechnen
r2 = r2_score(y, y_pred)
print(f'R²-Wert: {r2}')

# MSE berechnen
mse = mean_squared_error(y, y_pred)
print(f'Mittlerer quadratischer Fehler (MSE): {mse}')

# Residuen berechnen
residuals = y - y_pred

# Residuenplot
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Vorhergesagte Werte')
plt.ylabel('Residuen')
plt.title('Residuenplot')
plt.show()

# Vorhersagen mit dem Modell
y_pred = model.predict(x_poly)

# RMSE berechnen
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'Root Mean Square Error (RMSE) of the polynominal regression: {rmse}')

# MAE berechnen
mae = mean_absolute_error(y, y_pred)
print(f'Mean Absolute Error (MAE) of the polynominal regression: {mae}')

# Fit a linear regression model using statsmodels to get p-value
X_with_const = sm.add_constant(x_poly)  # Adding a constant for the intercept term
ols_model = sm.OLS(y, X_with_const)
ols_results = ols_model.fit()

# Print the summary statistics of the regression model
print(ols_results.summary())

x1 = np.array(max_leistung)
x2 = np.array(T_max)
y = np.array(investitionskosten)

# Kombinieren Sie x1 und x2 zu einem einzigen Array
X = np.column_stack((x1, x2))

# Konstante hinzufügen
X = sm.add_constant(X)
# Modell anpassen
model = sm.OLS(y, X).fit()

# R^2 und p-Wert des F-Tests
print(f"R-squared: {model.rsquared}")
print(f"p-Wert des F-Tests: {model.f_pvalue}")

# p-Werte der Koeffizienten
print(f"p-Werte der Koeffizienten: {model.pvalues}")

print('''In den OLS (Ordinary Least Squares) Regressionsergebnissen gibt es mehrere Schlüsselwerte, die für die Interpretation des Modells wichtig sind:
R-squared und Adj. R-squared

    R2R2 (R-squared): Dieser Wert beträgt 0,881, was bedeutet, dass das Modell etwa 88,1% der Varianz in der abhängigen Variable erklärt. Das ist ziemlich hoch und deutet darauf hin, dass das Modell gut passt.
    Adj. R2R2 (Adjusted R-squared): Dieser Wert beträgt 0,863 und ist eine angepasste Version von R2R2, die die Anzahl der Prädiktoren im Modell berücksichtigt. Auch dieser Wert ist hoch, was gut ist.

F-Statistik und Prob (F-statistic)

    F-Statistik: 48,24 ist der Wert der F-Statistik, die die Gesamtgüte des Modells testet.
    Prob (F-statistic): 9,66e-07 ist der p-Wert des F-Tests. Ein Wert nahe Null deutet darauf hin, dass das Modell signifikant besser ist als ein Modell ohne unabhängige Variablen. In diesem Fall ist der p-Wert extrem niedrig, was auf ein signifikantes Modell hindeutet.

Koeffizienten und ihre p-Werte (P>|t|)

    coef: Dies sind die Koeffizienten für die unabhängigen Variablen x1x1​ und x2x2​.
    P>|t|: Dies sind die p-Werte für die Hypothesentests, die die Nullhypothese testen, dass der jeweilige Koeffizient gleich Null ist. Beide sind kleiner als 0,05, was darauf hindeutet, dass beide Variablen signifikant sind.

Tests für Normalverteilung der Residuen

    Prob(Omnibus) und Prob(JB): Beide Tests prüfen die Normalverteilung der Residuen. Ein hoher p-Wert (> 0,05) deutet darauf hin, dass die Residuen normalverteilt sind. In diesem Fall sind die p-Werte 0,359 und 0,572, was darauf hindeutet, dass die Residuen wahrscheinlich normalverteilt sind.''')


