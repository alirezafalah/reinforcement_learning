import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs (replace with your file paths)
left_bound = pd.read_csv('rl_trajectory/yasBounds/LeftBound.csv', header=None, names=['x', 'y', 'z'])
right_bound = pd.read_csv('rl_trajectory/yasBounds/RightBound.csv', header=None, names=['x', 'y', 'z'])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(left_bound['x'], left_bound['y'], 'b-', label='Left Bound')
plt.plot(right_bound['x'], right_bound['y'], 'r-', label='Right Bound')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Yas Marina Track Bounds')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio to see track shape
plt.show()

# Print sample data
print("Left Bound Sample:\n", left_bound.head())
print("Right Bound Sample:\n", right_bound.head())