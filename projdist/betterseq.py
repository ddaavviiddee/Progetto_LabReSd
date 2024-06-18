import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from time import perf_counter

iris = load_iris()
X = iris.data
Y = iris.target

class_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

augmented_data = RandomOverSampler(sampling_strategy={0: 30000, 1: 30000, 2: 30000}, random_state=1)
X_resampled, Y_resampled = augmented_data.fit_resample(X, Y)

noise_factor = 0.2
X_resampled_noisy = X_resampled + noise_factor * np.random.randn(*X_resampled.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_resampled_noisy, Y_resampled, test_size=0.3, random_state=1)

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Numero di vicini
k = 3

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def euclidean_distance_matrix(a, b):
    return np.sqrt(np.sum((a[:, np.newaxis] - b) ** 2, axis=2))

def knn_classify(X_train, Y_train, X_test, k):
    prediction = []
    distances_matrix = euclidean_distance_matrix(X_test, X_train)
    print(distances_matrix)
    for _, distances in tqdm(enumerate(distances_matrix)):
        k_nearest_neighbors = np.argsort(distances)[:k]
        k_nearest_classes = Y_train[k_nearest_neighbors]
        most_common_class = Counter(k_nearest_classes).most_common(1)[0][0]
        prediction.append(most_common_class)\
    
    return prediction

start = perf_counter()
Y_pred = knn_classify(X_train_pca, Y_train, X_test_pca, k)
end = perf_counter()

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuratezza: {accuracy * 100:.2f}%, tempo: {end - start:.2f} secondi")

conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(Y_test, Y_pred, target_names=[class_names[i] for i in range(3)])
print("Classification Report:")
print(class_report)

colors = ['#cb91fc', '#85d77a', '#fc9191']

def plot_2d_density(X, Y, Y_pred=None, title=''):
    plt.figure(figsize=(10, 7))
    for class_value, color in zip(np.unique(Y), colors):
        sns.kdeplot(x=X[Y == class_value, 0], y=X[Y == class_value, 1], fill=True, color=color, label=class_names[class_value], alpha=0.5)
    if Y_pred is not None:
        plt.scatter(X[:, 0], X[:, 1], c=[colors[p] for p in Y_pred], marker='x', label='Predictions', alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.show()

plot_2d_density(X_train_pca[:, :2], Y_train, title='Distribuzione dei Dati di Addestramento (PCA)')
plot_2d_density(X_test_pca[:, :2], Y_test, Y_pred, 'Distribuzione dei Dati di Test e previsioni del KNN (PCA)')

def plot_3d_density(X, Y, Y_pred=None, title=''):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for class_value, color in zip(np.unique(Y), colors):
        indices = np.where(Y == class_value)
        ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], c=color, label=class_names[class_value], alpha=0.5, marker='o')
    if Y_pred is not None:
        for class_value, color in zip(np.unique(Y_pred), colors):
            indices = np.where(Y_pred == class_value)
            ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], c=color, marker='x', label=f'Predicted {class_names[class_value]}', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()

plot_3d_density(X_train_pca, Y_train, title='Distribuzione dei Dati di Addestramento (PCA)')
plot_3d_density(X_test_pca, Y_test, Y_pred, 'Distribuzione dei Dati di Test e previsioni del KNN (PCA)')
