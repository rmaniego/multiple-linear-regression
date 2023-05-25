import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Dataset
layers = [2439, 2198, 2227, 2017, 2056, 2184, 2301, 2255, 2325, 2187, 2112, 2270, 2004, 2072, 2013, 2432, 2072, 2436, 2394,
          2346, 2488, 2468, 2227, 2414, 2320, 2435, 2031, 2124, 2327, 2062, 2186, 2420, 2385, 2298, 2005, 2228, 2242, 2153,
          2007, 2329, 2367, 2491, 2118, 2276, 2227, 2219, 2128, 2247, 2082, 2433, 2354, 2160]
temperature = [25.8, 29.1, 24.4, 23.7, 31.1, 25.0, 26.7, 23.4, 28.9, 31.3, 26.4, 26.0, 24.3, 24.4, 28.6, 30.3, 23.2, 24.6,
               25.3, 29.6, 23.6, 25.8, 23.5, 25.4, 27.3, 23.5, 26.7, 27.9, 23.0, 25.7, 27.2, 23.5, 26.4, 30.2, 25.3, 31.5,
               29.7, 30.4, 30.7, 24.5, 25.5, 24.8, 27.6, 27.1, 23.1, 30.6, 25.4, 23.7, 25.6, 28.4, 30.8, 23.1]
egg_count = [1788, 1615, 1668, 1787, 1627, 1590, 1843, 2206, 1782, 1586, 1688, 1743, 1611, 1629, 1451, 1958, 2015, 1866,
             1891, 1699, 2404, 1932, 1985, 1905, 1800, 2147, 1514, 1590, 2278, 1519, 1711, 2160, 1774, 1767, 1557, 1695,
             1696, 1719, 1483, 1681, 1773, 1944, 1692, 1827, 2162, 1698, 1620, 2046, 1662, 1777, 1814, 2129]

# Convert the data to numpy arrays
X = np.array([layers, temperature]).T
y = np.array(egg_count)

# Fit the regression model
regression = LinearRegression()
regression.fit(X, y)

# Generate predictions
layers_pred = np.linspace(min(layers), max(layers), 10)
temperature_pred = np.linspace(min(temperature), max(temperature), 10)
layers_pred, temperature_pred = np.meshgrid(layers_pred, temperature_pred)
egg_count_pred = regression.predict(np.array([layers_pred.flatten(), temperature_pred.flatten()]).T)

# Plot the regression plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(layers, temperature, y, c='r', marker='o')
ax.plot_surface(layers_pred, temperature_pred, egg_count_pred.reshape(layers_pred.shape), alpha=0.5)
ax.set_xlabel('Layers')
ax.set_ylabel('Temperature')
ax.set_zlabel('Egg Count')
plt.show()
