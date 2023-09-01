import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# Simulierte Daten für ein 3D-Problem
np.random.seed(0)
x = np.random.rand(50)
y = np.random.rand(50)
z = 2 * x + 3 * y + 5 + np.random.normal(0, 0.1, 50)

# Definition der Funktion, die die Kurve beschreibt
def curve_function(x, y, a, b, c):
    return a * x + b * y + c

# Verwende curve_fit, um die Funktion an die Daten anzupassen
popt, _ = curve_fit(curve_function, (x, y), z)

a_fit, b_fit, c_fit = popt

# Erstelle ein 3D-Plot der Daten und der angepassten Kurve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot der simulierten Datenpunkte
ax.scatter(x, y, z, c='blue', marker='o', label='Datenpunkte')

# Erstelle ein Raster von x- und y-Werten für die angepasste Kurve
x_fit = np.linspace(min(x), max(x), 50)
y_fit = np.linspace(min(y), max(y), 50)
X_fit, Y_fit = np.meshgrid(x_fit, y_fit)

# Berechne die z-Werte der angepassten Kurve basierend auf den x- und y-Werten
Z_fit = curve_function(X_fit, Y_fit, a_fit, b_fit, c_fit)

# Plot der angepassten Kurve
ax.plot_surface(X_fit, Y_fit, Z_fit, cmap='viridis', alpha=0.5, label='Angepasste Kurve')

ax.set_xlabel('X-Achse')
ax.set_ylabel('Y-Achse')
ax.set_zlabel('Z-Achse')

plt.legend()
plt.show()
