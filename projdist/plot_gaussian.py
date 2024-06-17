import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np

iris = load_iris()
X = iris.data
Y = iris.target

class_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

augmented_data = RandomOverSampler(sampling_strategy={0: 1000, 1: 1000, 2: 1000}, random_state=1)
X_resampled, Y_resampled = augmented_data.fit_resample(X, Y)

noise_factor = 0.2
X_resampled_noisy = X_resampled + noise_factor * np.random.randn(*X_resampled.shape)

plt.figure(figsize=(10, 6))

plt.subplot(121)
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=Y_resampled, cmap='viridis', label='Originali')

plt.subplot(122)
plt.scatter(X_resampled_noisy[:, 0], X_resampled_noisy[:, 1], c=Y_resampled, cmap='viridis', label='Con rumore')

plt.xlabel('Prima feature')
plt.ylabel('Seconda feature')
plt.suptitle('Iris dataset - Prima e dopo rumore gaussiano (Features 1 e 2)')
plt.legend()
plt.show()
