from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Carregamento da base de dados 
iris = datasets.load_iris()
# visualização de quantos registros existem por classe
#np.unique: retorna não apenas os valores únicos do array, 
# mas também a quantidade de vezes que cada valor único aparece no array.
unicos, quantidade = np.unique(iris.target, return_counts = True)
print(f'Unicos: {unicos},\nQuantidade: {quantidade}')

# Agrupamento com k-means, utilizando 3 clusters (de acordo com a base de dados)
# n_clusters: número de clusters (ou grupos) que o algoritmo K-Means irá tentar formar a partir dos dados
cluster = KMeans(n_clusters = 3)
cluster.fit(iris.data)

# Visualização dos três centroides
# acessa os centroides dos clusters encontrados pelo modelo KMeans após o treinamento.
centroides = cluster.cluster_centers_
print(f'Centroides: {centroides}')

# Visualização dos grupos que cada registro foi associado
agrupamento = cluster.labels_
print(f'Previsões: {agrupamento}')

# Contagem dos registros por classe
unicos2, quantidade2 = np.unique(agrupamento, return_counts = True)
print(f'Unicos: {unicos2},\nQuantidade: {quantidade2}')

# Geração da matriz de confusão para comparar os grupos com a base de dados
resultados = confusion_matrix(iris.target, agrupamento)
print(resultados)

# Geração do gráfico com os clusters gerados, considerando para um (agrupamento 0, 1 ou 2)
# Usamos somente as colunas 0 e 1 da base de dados original para termos 2 dimensões
plt.scatter(iris.data[agrupamento == 0, 0], iris.data[agrupamento == 0, 1], 
            c = 'green', label = 'Setosa')
plt.scatter(iris.data[agrupamento == 1, 0], iris.data[agrupamento == 1, 1], 
            c = 'red', label = 'Versicolor')
plt.scatter(iris.data[agrupamento == 2, 0], iris.data[agrupamento == 2, 1], 
            c = 'blue', label = 'Virgica')
plt.legend()

plt.show()