import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import os
import numpy as np
from pygments.lexers.objective import objective
from scipy.optimize import curve_fit
from scipy.interpolate import SmoothBivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
ax.set_yticks(range(0, 1001, 100))  # Hier werden Ticks bei 0, 10, 20, ..., 100 gesetzt
ax.set_zticks(range(0, 201, 20))  # Hier werden Ticks bei 0, 10, 20, ..., 100 gesetzt

ax.set_xlabel('max. Leistung')
ax.set_ylabel('Spezifische Investitionskosten [€/kW]')
ax.set_zlabel('Senke-Aus [°C]')

# Diagramm anzeigen
#plt.show()

# Den aktuellen Pfad der Python-Datei abrufen
current_directory = os.path.dirname(os.path.abspath(__file__))

'''curve fit-Funktion zur Berechnung der Kurve'''
x1_data = df['max. Leistung'].values  # verwandel die Spalte in ein Array
x2_data = df['Senke-Aus'].values
y_data = df['Spezifische Invest (€/kW) 2023'].values
def WP_curve(x, a, b, c,d,e):
    x1, x2 = x
    return a * (b - x1) + c * x1*x2 + d*(x2**3) + e
# Neue Funktionen
def fit_homo_linear(x, a):
    x = x1, x2
    return a*(x1*x2)

def fit_homo_linear2(x1, x2, a, b):
    x1, x2 = x
    return a * x1 + b * x2

def fit_linear(x, a, b):
    x = np.array(x)
    return a*x + b

def fit_power(x, a, b):
    x = np.array(x)
    return a*x**b

def fit_log(x, a):
    x = np.array(x)
    return np.log(x) / np.log(a)

def fit_quadratic(x, a, b, c):
    x = np.array(x)
    return a*x**2 + b * x + c

# Funktion, um R-Quadrat zu berechnen
def r_squared(popt, f, x, y):
    f_x = f(x, *popt)  # Unpacking der Parameter
    residuals = y - f_x
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Eine Liste von zu prüfenden Funktionen und ihrer Parameter
functions_to_fit = [
    {'func': WP_curve, 'params': 5},
    {'func': fit_homo_linear, 'params': 1},
    {'func': fit_linear, 'params': 2},
    {'func': fit_power, 'params': 2},
    {'func': fit_log, 'params': 1},
    {'func': fit_quadratic, 'params': 3},
]

# Anpassung für jede Funktion
for func_dict in functions_to_fit:
    func = func_dict['func']
    params = func_dict['params']

    popt, _ = curve_fit(func, x1_data, y_data, maxfev=10000)  # Parameter anpassen
    r2 = r_squared(popt, func, x1_data, y_data)  # R-Quadrat berechnen

    print(f"Funktion: {func.__name__}, R-Quadrat: {r2}, Parameter: {popt}")

# Beispiel-Funktionen
def func1(x, a, b, c):
    x1, x2 = x
    return a * x1 + b * x2 + c

def func2(x, a, b, c, d):
    x1, x2 = x
    return a * x1**2 + b * x2**2 + c * x1 + d

# Funktion, um R-Quadrat zu berechnen
def r_squared(y_actual, y_pred):
    residuals = y_actual - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

# Liste der Funktionen
functions = [func1, func2]

for func in functions:
    # Curve Fit
    popt, _ = curve_fit(func, (x1_data, x2_data), y_data, maxfev=10000)
    # Berechne den vorhergesagten y
    y_pred = func((x1_data, x2_data), *popt)
    # R-Quadrat
    r2 = r_squared(y_data, y_pred)
    print(f"Funktion: {func.__name__}, R-Quadrat: {r2}, Parameter: {popt}")


# '''Ende des versuchs curvefit zu implementieren'''
print(len(x1_data), len(x2_data), len(y_data))

# choose the input and output variables
x1=x1_data
x2=x2_data
# curve fit
popt, pcov = curve_fit(WP_curve, (x1, x2), y_data,p0=[1, 1, 1, 1, 1])
# summarize the parameter values
a, b, c, d, e = popt
print('popt:',popt)
print('pcov:',pcov)
# plot input vs output
plt.scatter(x1_data, x2_data, y_data)
# define a sequence of inputs between the smallest and largest known inputs

x1_data= np.linspace(min(x1_data), max(x1_data), 100)
x2_data= np.linspace(min(x2_data), max(x2_data), 100)
# calculate the output for the range
y_data = WP_curve((x1,x2), a, b, c, d, e)
# create a line plot for the mapping function
plt.plot(x1_data, x2_data, y_data, color='red')
plt.show()


#alternative:
#now including the optimal parameters in the plot
# a_opt, b_opt, c_opt = popt
# x_data= np.linspace(min(x_data), max(x_data), 100)
# y_data = EK_f(x_data, a_opt, b_opt, c_opt)
# #plot the data with the curve fit
# plt.scatter(x_data, y_data, color='r', linewidths=0.1)
# plt.show()
# #plot the covariance matrix
# plt.imshow(np.log(np.abs(pcov)))
# plt.colorbar()
# plt.show()

'''Spline fitting'''
#mit spline fitting:
# x = np.array(max_leistung)
# y = np.array(investitionskosten)
# z = np.array(T_max)
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

# '''Polynomial fitting'''
# x1 = T_max.values.reshape(-1, 1)
# x2 = max_leistung.values.reshape(-1, 1)
# y = investitionskosten.values
#
# # Choose the degrees of the polynomials
# degree_x1 = 3
# degree_x2 = 3
#
# # Transform the data to include polynomial features
# poly_features = PolynomialFeatures(degree=[degree_x1, degree_x2])
# x_poly = poly_features.fit_transform(np.hstack((x1, x2)))
#
# # Fit a linear regression model to the polynomial features
# model = LinearRegression()
# model.fit(x_poly, y)
#
# '''Funktionsterm ausgeben'''
# # Die Koeffizienten des Polynommodells abrufen
# coefficients = model.coef_
# intercept = model.intercept_
#
# # Erstelle die Namen der Polynomfeatures manuell
# poly_feature_names = []
# for i in range(degree_x1 + 1):
#     for j in range(degree_x2 + 1):
#         if i + j > 0:
#             term = f'x1^{i} * x2^{j}'
#             poly_feature_names.append(term)
#
# # Den Funktionsterm des Polynommodells erstellen
# function_term = f'{intercept:.2f} + '
# function_term += ' + '.join([f'{coef:.2f} * {poly_feature_names[i]}' for i, coef in enumerate(coefficients)])
#
# print(f'Funktionsterm des Polynommodells: {function_term}')
#
#
# '''Plotting the surface'''
#
# # Generating a meshgrid of values for plotting
# x1_values = np.linspace(min(x1), max(x1), 100)
# x2_values = np.linspace(min(x2), max(x2), 100)
# x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
#
# # Transform the meshgrid to polynomial features
# x_grid_poly = poly_features.transform(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
#
# # Predict investment costs using the model
# predicted_invest_costs = model.predict(x_grid_poly)
#
# # Reshape the predictions back to the grid shape
# predicted_invest_costs_grid = predicted_invest_costs.reshape(x1_grid.shape)
#
# # Plotting the surface
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_surface(x1_grid, x2_grid, predicted_invest_costs_grid, cmap='viridis', alpha=0.5)
#
# ax.scatter(x1, x2, y, color='blue', label='Datenpunkte')
#
# ax.set_xlabel('Senke-Aus [°C]')
# ax.set_ylabel('Maximale Leistung [MW]')
# ax.set_zlabel('Spezifische Investitionskosten [€/kW]')
#
# plt.show()