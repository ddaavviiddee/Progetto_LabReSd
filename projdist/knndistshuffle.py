import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from time import perf_counter
import ray
import seaborn as sns
from tqdm import tqdm
import gc

ray.init(ignore_reinit_error=True)

# Caricamento del dataset Iris
iris = load_iris()
X = iris.data
Y = iris.target

# Assegnazione dei class names
class_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Data augmentation con RandomOverSampler
augmented_data = RandomOverSampler(sampling_strategy={0: 40000, 1: 40000, 2: 40000}, random_state=1)
X_resampled, Y_resampled = augmented_data.fit_resample(X, Y)

# Aggiunta di rumore gaussiano sul dataset
noise_factor = 0.2
X_resampled_noisy = X_resampled + noise_factor * np.random.randn(*X_resampled.shape)

# Suddivisione del dataset in training e test set (70% train, 30% test)
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled_noisy, Y_resampled, test_size=0.3, random_state=1)

# Riduzione dimensionale con PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

def euclidean_distance_matrix(a, b):
    return np.sqrt(np.sum((a[:, np.newaxis] - b) ** 2, axis=2))

@ray.remote
def knn_mapper(X_train, Y_train, X_test_partition, partition_index, k):
    # Ogni mapper riceve una partizione di dati, e su di questa calcola 
    # la matrice delle distanze
    distances_matrix = euclidean_distance_matrix(X_test_partition, X_train)
    mapper_results = []
    # Calcola per ogni punto le k classi più vicine
    for i, distances in enumerate(distances_matrix):
        k_nearest_neighbors = np.argsort(distances)[:k]
        k_nearest_classes = Y_train[k_nearest_neighbors]
        mapper_results.append((partition_index + i, k_nearest_classes))
        gc.collect()
    # Risultato (partition_idx + i, array[k_n classes])
    return mapper_results

@ray.remote
def knn_reducer(mapper_results_partition):
    predictions = []
    # I reducer prendono una parte dei dati dei mapper
    # e calcolano la classe più comune da assegnare al punto
    for i, k_nearest_classes in mapper_results_partition:
        most_common_class = Counter(k_nearest_classes).most_common(1)[0][0]
        predictions.append((i, most_common_class))
        gc.collect()
    # Risultato (i (ovvero l'idx del punto), most_common_class (0 oppure 1 oppure 2))
    return predictions

def knn(k, num_mappers, num_reducers):
    # Split dei dati per i mapper
    X_test_partitions = np.array_split(X_test_pca, num_mappers)

    start = perf_counter()

    # Assegnazione dei dati ai mapper
    mapper_futures = [knn_mapper.remote(X_train_pca, Y_train, partition, i * len(partition), k) for i, partition in enumerate(X_test_partitions)]
    
    mapper_results = []
    print("Estraendo i dati dai mapper...")
    for future in tqdm(ray.get(mapper_futures), desc="Mappatura"):
        mapper_results.extend(future)

    # Shuffling 
    np.random.shuffle(mapper_results)
    reducer_data = [[] for _ in range(num_reducers)]
    for idx, item in enumerate(mapper_results):
        reducer_data[idx % num_reducers].append(item)

    # Assegnazione dei dati ai reducer
    reducer_futures = [knn_reducer.remote(chunk) for chunk in reducer_data]
    print("Estraendo i dati dai reducer...")
    reducer_results = ray.get(reducer_futures)
    
    # Bundling dei risultati dei reducer
    predictions = [0] * len(X_test_pca)
    for reducer_result in reducer_results:
        for i, pred in reducer_result:
            predictions[i] = pred

    accuracy = accuracy_score(Y_test, predictions)
    end = perf_counter()
    print(f"Precisione: {accuracy * 100:.2f}%, tempo: {end - start:.2f} secondi")
    
    conf_matrix = confusion_matrix(Y_test, predictions)
    print("Matrice di confusione:")
    print(conf_matrix)
    
    class_report = classification_report(Y_test, predictions, target_names=[class_names[i] for i in range(3)])
    print("Dati aggiuntivi:")
    print(class_report)

    return predictions, accuracy, conf_matrix

k = 3
num_mappers = 24
num_reducers = 24
Y_pred, accuracy, conf_matrix = knn(k, num_mappers, num_reducers)

ray.shutdown()

colors = ['#cb91fc', '#85d77a', '#fc9191']

# Plot
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

def plot_confusion_matrix_heatmap(conf_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap')
    plt.show()

plot_confusion_matrix_heatmap(conf_matrix, [class_names[i] for i in range(3)])
