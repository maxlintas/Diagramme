import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import matplotlib.pyplot as plt

# Datenpunkte vorbereiten
temperature = np.array([...])  # Temperaturwerte
max_power = np.array([...])     # Maximale Leistungswerte
invest_costs = np.array([...])  # Investitionskosten

# Spline-Fitting durchführen
spline = SmoothBivariateSpline(temperature, max_power, invest_costs, kx=3, ky=3)

# Neue Werte für Vorhersage
new_temperature = np.linspace(min(temperature), max(temperature), 100)
new_max_power = np.linspace(min(max_power), max(max_power), 100)

# Meshgrid erstellen
X, Y = np.meshgrid(new_temperature, new_max_power)

# Geschätzte Investitionskosten für das Meshgrid
predicted_invest_costs = spline(X, Y)

# 3D-Diagramm anzeigen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, predicted_invest_costs, cmap='viridis')

ax.set_xlabel('Temperatur')
ax.set_ylabel('Maximale Leistung')
ax.set_zlabel('Investitionskosten')
plt.show()