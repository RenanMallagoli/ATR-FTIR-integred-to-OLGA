import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

def build_knn(X_total, k=3):
    """Constrói o Grafo de k-Vizinhos Mais Próximos (KNNG) tradicional."""
    A = kneighbors_graph(X_total, n_neighbors=k, mode='connectivity', include_self=False).toarray()
    return np.maximum(A, A.T)

def build_sknn(X_total, k=3):
    """Constrói o Grafo k-NN Seletivo (S-KNNG)."""
    A = np.zeros((len(X_total), len(X_total)), dtype=int)
    A_controle = build_knn(X_total[:50], k)
    A_anomalia = build_knn(X_total[50:], k)
    A[0:50, 0:50] = A_controle
    A[50:60, 50:60] = A_anomalia
    return A

def build_mknn(X_total, k=3):
    """
    Constrói o Grafo k-NN Mútuo (M-KNNG).
    Aresta só existe se i é vizinho de j E j é vizinho de i.
    """
    A = kneighbors_graph(X_total, n_neighbors=k, mode='connectivity', include_self=False).toarray()
    # Em vez de np.maximum (que pega a união), usamos np.minimum (que pega a interseção)
    return np.minimum(A, A.T)

def build_smknn(X_total, k=3):
    """Constrói o Grafo k-NN Mútuo Seletivo (S-MKNNG)."""
    A = np.zeros((len(X_total), len(X_total)), dtype=int)
    A_controle = build_mknn(X_total[:50], k)
    A_anomalia = build_mknn(X_total[50:], k)
    A[0:50, 0:50] = A_controle
    A[50:60, 50:60] = A_anomalia
    return A

def build_cosine_knn(X_total, k=3):
    """
    Grafo k-NN baseado na Similaridade por Cosseno.
    Foca no 'ângulo' (assinatura espectral) e ignora a intensidade bruta.
    """
    # Calcula a matriz de similaridade [-1 a 1]. Quanto maior, mais similar.
    sim_matrix = cosine_similarity(X_total)
    
    # Preenche a diagonal com infinito negativo para o nó não se conectar a si mesmo
    np.fill_diagonal(sim_matrix, -np.inf) 
    
    A = np.zeros_like(sim_matrix, dtype=int)
    for i in range(len(X_total)):
        # np.argsort ordena de forma crescente. Pegamos os k últimos (maiores valores)
        top_k_indices = np.argsort(sim_matrix[i])[-k:]
        A[i, top_k_indices] = 1
        
    # Garante que a matriz seja simétrica (grafo não direcionado)
    return np.maximum(A, A.T)


def build_pearson_network(X_total, threshold=0.95):
    """
    Rede de Correlação de Pearson.
    Conecta nós cujas tendências de absorção variam juntas.
    Não usa k-vizinhos, mas sim um limiar de força de correlação.
    """
    # np.corrcoef retorna valores de -1 (correlação inversa) a 1 (correlação perfeita)
    corr_matrix = np.corrcoef(X_total)
    
    # Preenche a diagonal com 0 para evitar auto-conexões
    np.fill_diagonal(corr_matrix, 0)
    
    # Cria aresta apenas se a correlação for maior que o limiar estipulado
    A = (corr_matrix > threshold).astype(int)
    
    # A matriz de correlação já é simétrica por natureza matemática
    return A


def build_dtw_knn(X_total, k=3):
    """
    Grafo k-NN baseado em Dynamic Time Warping (DTW).
    Alinha os picos elásticos da série temporal.
    """
    try:
        from fastdtw import fastdtw
    except ImportError:
        raise ImportError("A biblioteca fastdtw não está instalada. Rode: pip install fastdtw")

    n_samples = len(X_total)
    dist_matrix = np.zeros((n_samples, n_samples))
    
    # Calculando a matriz de distância DTW par a par
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Usando uma função lambda simples para a distância escalar
            distance, _ = fastdtw(X_total[i], X_total[j], dist=lambda a, b: abs(a - b), radius=10)
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance 
            
    np.fill_diagonal(dist_matrix, np.inf)
    
    A = np.zeros_like(dist_matrix, dtype=int)
    for i in range(n_samples):
        top_k_indices = np.argsort(dist_matrix[i])[:k] 
        A[i, top_k_indices] = 1
        
    return np.maximum(A, A.T)

def build_visg2(X_total, k=3):
    """
    Constrói o Meta-Grafo baseado no VisG2 (Abordagem Não-Supervisionada por Jaccard k-NN).
    Calcula a similaridade de Jaccard dos Grafos de Visibilidade e conecta os k maiores valores.
    """
    try:
        from ts2vg import NaturalVG
    except ImportError:
        raise ImportError("A biblioteca ts2vg não está instalada. Rode: pip install ts2vg")

    n_samples = len(X_total)
    
    # 1. Construção do NVG e extração do conjunto de arestas
    edge_sets = []
    for i in range(n_samples):
        vg = NaturalVG()
        vg.build(X_total[i])
        edge_sets.append(set(vg.edges))
        
    # 2. Matriz de Similaridade de Jaccard Topológica
    sim_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            intersecao = len(edge_sets[i].intersection(edge_sets[j]))
            uniao = len(edge_sets[i].union(edge_sets[j]))
            jaccard = intersecao / uniao if uniao != 0 else 0
            
            sim_matrix[i, j] = jaccard
            sim_matrix[j, i] = jaccard
            
    # Zera a diagonal para evitar auto-conexões
    np.fill_diagonal(sim_matrix, -1)
    
    # 3. Formação do Meta-Grafo (k-NN não-supervisionado baseado em Jaccard)
    A = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        # Pega os k maiores índices de Jaccard para cada espectro, cegamente
        top_k_indices = np.argsort(sim_matrix[i])[-k:]
        A[i, top_k_indices] = 1
        
    # Garante a simetria do grafo
    return np.maximum(A, A.T)