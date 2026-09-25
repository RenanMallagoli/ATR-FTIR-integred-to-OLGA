import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

def generate_ftir_spectra(n_samples=100, is_anomaly=False, random_seed=42):
    np.random.seed(random_seed)
    
    # Eixo X: Números de onda (truncado entre 1800 e 900 cm^-1 para focar nas regiões de interesse)
    wavenumbers = np.linspace(1800, 900, 500)
    
    # Definição dos picos base (Amida I, Amida II, etc.)
    # Formato: (Amplitude base, Posição mu, Largura sigma)
    base_peaks = [
        (0.8, 1650, 20),
        (0.5, 1550, 25),
        (0.3, 1050, 15)
    ]
    
    spectra = []
    
    for _ in range(n_samples):
        y = np.zeros_like(wavenumbers)
        for A, mu, sigma in base_peaks:
            # Variação natural para cada amostra
            A_var = A + np.random.normal(0, 0.05)
            
            # Se for anomalia, introduzimos uma perturbação maior na amplitude e posição
            if is_anomaly:
                A_var *= np.random.uniform(0.5, 1.5)
                mu += np.random.normal(0, 5)
                
            # Equação Gaussiana
            y += A_var * np.exp(-((wavenumbers - mu)**2) / (2 * sigma**2))
            
        # Adição do ruído branco do sensor
        y += np.random.normal(0, 0.02, size=wavenumbers.shape)
        spectra.append(y)
        
    return wavenumbers, np.array(spectra)

# Gerando 50 amostras da classe de interesse e 10 anomalias
wavenumbers, X_normal = generate_ftir_spectra(n_samples=50, is_anomaly=False, random_seed=1)
_, X_anomaly = generate_ftir_spectra(n_samples=10, is_anomaly=True, random_seed=2)

print(f"Shape dos dados normais: {X_normal.shape}")

plt.figure(figsize=(10, 6))

# Plotando 10 amostras da classe de interesse (Controle) em azul
for i in range(10):
    plt.plot(wavenumbers, X_normal[i], color='blue', alpha=0.6, 
             label='Controle (Interesse)' if i == 0 else "")

# Plotando as 10 amostras da classe de anomalia em vermelho
for i in range(10):
    plt.plot(wavenumbers, X_anomaly[i], color='red', alpha=0.6, 
             label='Anomalia' if i == 0 else "")

plt.gca().invert_xaxis() # Invertendo o eixo X para o padrão FTIR
plt.xlabel("Número de Onda (cm⁻¹)")
plt.ylabel("Intensidade")
plt.title("Espectros FTIR Simulados: Controle vs Anomalia")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 1. Preparando a Matriz de Nós (V)
# Vamos unir os 50 controles e as 10 anomalias em um único dataset
X_total = np.vstack((X_normal, X_anomaly))

# Criando um array de rótulos (labels) apenas para nossa avaliação posterior
# 0 = Controle (Classe de Interesse), 1 = Anomalia
y_total = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

# 2. Construindo a Matriz de Adjacência (A) via k-NN
k_vizinhos = 3

# O kneighbors_graph calcula a distância Euclidiana de todos contra todos 
# e retorna 1 para os 'k' mais próximos.
A_knn_sparse = kneighbors_graph(X_total, n_neighbors=k_vizinhos, mode='connectivity', include_self=False)

# Convertendo para uma matriz densa para facilitar a manipulação
A_knn = A_knn_sparse.toarray()

# Tornando o grafo não-direcionado (se A é vizinho de B, B é vizinho de A)
A_knn = np.maximum(A_knn, A_knn.T)

print(f"Formato da Matriz de Representação dos Nós (V): {X_total.shape}")
print(f"Formato da Matriz de Adjacência (A): {A_knn.shape}")
print(f"Total de arestas criadas no grafo k-NN: {np.sum(A_knn) // 2}")

# Visualizando a Matriz de Adjacência gerada
plt.figure(figsize=(6, 6))
plt.imshow(A_knn, cmap='Greys', interpolation='none')
plt.title(f"Matriz de Adjacência (Grafo k-NN, k={k_vizinhos})")
plt.xlabel("Índice do Nó (Espectro)")
plt.ylabel("Índice do Nó (Espectro)")
plt.show()