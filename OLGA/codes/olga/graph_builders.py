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

def _prune_nvg_edges(canonical_edges, nvg_mode="none", nvg_param=None, neighbor_policy="closest_band_union"):
    """
    Aplica a restrição de esparsidade/vizinhança (Decisão 01) sobre o conjunto
    canônico de arestas não-direcionadas {(u, v) com u < v} de um NVG.
    """
    valid_modes = ("none", "neighbor_limit", "band_window")
    if nvg_mode not in valid_modes:
        raise ValueError(f"nvg_mode inválido: '{nvg_mode}'. Escolha entre {valid_modes}.")

    if nvg_mode == "none":
        return canonical_edges

    if nvg_param is None or isinstance(nvg_param, bool) or int(nvg_param) < 1:
        raise ValueError(
            f"Para nvg_mode='{nvg_mode}', informe um inteiro positivo em nvg_param (recebido: {nvg_param})."
        )
    limit = int(nvg_param)

    if nvg_mode == "band_window":
        return {(u, v) for (u, v) in canonical_edges if (v - u) <= limit}

    valid_policies = ("closest_band_union", "right_n")
    if neighbor_policy not in valid_policies:
        raise ValueError(
            f"neighbor_policy inválida: '{neighbor_policy}'. Escolha entre {valid_policies}."
        )

    if neighbor_policy == "closest_band_union":
        adj = {}
        for u, v in canonical_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        pruned = set()
        for a, neighbors in adj.items():
            ordered = sorted(neighbors, key=lambda b: (abs(b - a), b))
            for b in ordered[:limit]:
                pruned.add((a, b) if a < b else (b, a))
        return pruned

    # neighbor_policy == "right_n": cada nó u mantém até os primeiros n visíveis à direita (v > u)
    right_adj = {}
    for u, v in canonical_edges:
        right_adj.setdefault(u, []).append(v)

    pruned = set()
    for u, right_neighbors in right_adj.items():
        for v in sorted(right_neighbors)[:limit]:
            pruned.add((u, v))
    return pruned


def _extract_nvg_edges(
    spectrum,
    nvg_mode="none",
    nvg_param=None,
    neighbor_policy="closest_band_union",
    vg_cls=None,
):
    """Constrói o NVG de um espectro e retorna seu conjunto canônico de arestas."""
    if vg_cls is None:
        try:
            from ts2vg import NaturalVG
            vg_cls = NaturalVG
        except ImportError:
            raise ImportError("A biblioteca ts2vg não está instalada. Rode: pip install ts2vg")

    vg = vg_cls()
    vg.build(spectrum)
    raw_edges = getattr(vg, "edges_unweighted", vg.edges)
    canonical_edges = {
        (int(u), int(v)) if u < v else (int(v), int(u))
        for u, v, *_ in raw_edges
        if u != v
    }
    return _prune_nvg_edges(
        canonical_edges,
        nvg_mode=nvg_mode,
        nvg_param=nvg_param,
        neighbor_policy=neighbor_policy,
    )


def _compute_jaccard_matrix(edge_sets):
    """Calcula a matriz simétrica de similaridade de Jaccard sobre conjuntos de arestas (Eqs. 2–3)."""
    n_samples = len(edge_sets)
    sizes = [len(es) for es in edge_sets]
    sim_matrix = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        ei = edge_sets[i]
        si = sizes[i]
        for j in range(i + 1, n_samples):
            intersecao = len(ei.intersection(edge_sets[j]))
            uniao = si + sizes[j] - intersecao
            jaccard = intersecao / uniao if uniao != 0 else 0.0
            sim_matrix[i, j] = jaccard
            sim_matrix[j, i] = jaccard

    return sim_matrix


def _build_topk_metagraph(sim_matrix, k=3, meta_mode="unsupervised", y=None, train_mask=None):
    """
    Constrói a matriz binária simétrica de adjacência do meta-grafo VisG2 (Decisão 02).
    """
    n_samples = sim_matrix.shape[0]
    if isinstance(k, bool) or int(k) < 1 or int(k) >= n_samples:
        raise ValueError(f"O parâmetro k deve satisfazer 1 <= k < {n_samples} (recebido: {k}).")
    k = int(k)

    valid_meta_modes = ("unsupervised", "train_only", "train_supervised")
    if meta_mode not in valid_meta_modes:
        raise ValueError(
            f"meta_mode inválido: '{meta_mode}'. Escolha entre {valid_meta_modes}."
        )

    scores = sim_matrix.astype(float, copy=True)
    np.fill_diagonal(scores, -np.inf)

    if meta_mode in ("train_only", "train_supervised"):
        if train_mask is None:
            raise ValueError(f"meta_mode='{meta_mode}' exige train_mask explícita.")
        train_mask_bool = np.asarray(train_mask, dtype=bool)
        if train_mask_bool.shape != (n_samples,):
            raise ValueError("train_mask deve ter o mesmo comprimento de X_total.")
        if int(np.sum(train_mask_bool)) <= 1:
            raise ValueError("train_mask deve conter ao menos 2 amostras de treino.")

        # Impede que nós fora de train_mask sejam candidatos na seleção dirigida top-k
        scores[:, ~train_mask_bool] = -np.inf

        if meta_mode == "train_supervised":
            if y is None:
                raise ValueError("meta_mode='train_supervised' exige o vetor y.")
            y_arr = np.asarray(y)
            if y_arr.shape != (n_samples,):
                raise ValueError("y deve ter o mesmo comprimento de X_total.")

            # Lê estritamente os rótulos dentro de train_mask (zero acesso a val/test)
            train_indices = np.where(train_mask_bool)[0]
            y_train_only = {int(idx): y_arr[idx] for idx in train_indices}
            for i in train_indices:
                yi = y_train_only[int(i)]
                for j in train_indices:
                    if i != j and y_train_only[int(j)] != yi:
                        scores[i, j] = -np.inf

    A = np.zeros((n_samples, n_samples), dtype=int)
    for i in range(n_samples):
        order = np.argsort(scores[i], kind="stable")
        top_candidates = [idx for idx in order[-k:] if np.isfinite(scores[i, idx])]
        if top_candidates:
            A[i, top_candidates] = 1

    return np.maximum(A, A.T)


def _count_connected_components(A):
    """Conta componentes conexas de uma matriz de adjacência simétrica sem dependências externas."""
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = 0
    for start in range(n):
        if not visited[start]:
            components += 1
            stack = [start]
            visited[start] = True
            while stack:
                u = stack.pop()
                for v in np.where(A[u] > 0)[0]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(int(v))
    return components


def compute_visg2_diagnostics(
    edge_sets,
    sim_matrix,
    A,
    y=None,
    train_mask=None,
    val_mask=None,
    test_mask=None,
):
    """
    Calcula métricas estruturais de auditoria do VisG2 em modo estritamente somente-leitura
    (não altera o grafo nem injeta informação de rótulos no modelo).
    """
    n_samples = len(edge_sets)
    nvg_sizes = np.array([len(es) for es in edge_sets], dtype=float)
    tri_u = sim_matrix[np.triu_indices(n_samples, k=1)]
    degrees = np.sum(A > 0, axis=1)
    edge_u, edge_v = np.where(np.triu(A, k=1) > 0)
    n_meta_edges = len(edge_u)

    diag = {
        "n_samples": n_samples,
        "nvg_edges_mean": float(np.mean(nvg_sizes)) if n_samples else 0.0,
        "nvg_edges_std": float(np.std(nvg_sizes)) if n_samples else 0.0,
        "nvg_edges_min": int(np.min(nvg_sizes)) if n_samples else 0,
        "nvg_edges_max": int(np.max(nvg_sizes)) if n_samples else 0,
        "jaccard_mean": float(np.mean(tri_u)) if len(tri_u) else 0.0,
        "jaccard_std": float(np.std(tri_u)) if len(tri_u) else 0.0,
        "jaccard_median": float(np.median(tri_u)) if len(tri_u) else 0.0,
        "jaccard_max": float(np.max(tri_u)) if len(tri_u) else 0.0,
        "jaccard_zero_pct": float(100.0 * np.mean(tri_u == 0.0)) if len(tri_u) else 0.0,
        "meta_edges": int(n_meta_edges),
        "meta_deg_mean": float(np.mean(degrees)) if n_samples else 0.0,
        "meta_deg_min": int(np.min(degrees)) if n_samples else 0,
        "meta_deg_max": int(np.max(degrees)) if n_samples else 0,
        "meta_components": int(_count_connected_components(A)) if n_samples else 0,
    }

    if train_mask is not None and n_meta_edges > 0:
        tm = np.asarray(train_mask, dtype=bool)
        u_tr, v_tr = tm[edge_u], tm[edge_v]
        both_train = u_tr & v_tr
        cross_train = u_tr ^ v_tr
        both_unlabeled = (~u_tr) & (~v_tr)
        diag["pct_edges_train_train"] = float(100.0 * np.mean(both_train))
        diag["pct_edges_train_unlabeled"] = float(100.0 * np.mean(cross_train))
        diag["pct_edges_unlabeled_unlabeled"] = float(100.0 * np.mean(both_unlabeled))

        if val_mask is not None and test_mask is not None:
            vm = np.asarray(val_mask, dtype=bool)
            tsm = np.asarray(test_mask, dtype=bool)
            train_val = (tm[edge_u] & vm[edge_v]) | (vm[edge_u] & tm[edge_v])
            train_test = (tm[edge_u] & tsm[edge_v]) | (tsm[edge_u] & tm[edge_v])
            diag["pct_edges_train_val"] = float(100.0 * np.mean(train_val))
            diag["pct_edges_train_test"] = float(100.0 * np.mean(train_test))

    if y is not None and n_meta_edges > 0:
        y_arr = np.asarray(y)
        diag["homophily_pct"] = float(100.0 * np.mean(y_arr[edge_u] == y_arr[edge_v]))

    return diag


def format_visg2_diagnostics(diag):
    """Formata o dicionário de diagnóstico do VisG2 para registro legível no console."""
    lines = [
        "[VisG2 Diagnóstico Estrutural]",
        (
            f"  NVG/espectro  : arestas = {diag['nvg_edges_mean']:.1f} ± {diag['nvg_edges_std']:.1f} "
            f"(min={diag['nvg_edges_min']}, max={diag['nvg_edges_max']})"
        ),
        (
            f"  Jaccard       : média = {diag['jaccard_mean']:.4f} ± {diag['jaccard_std']:.4f} | "
            f"mediana = {diag['jaccard_median']:.4f} | max = {diag['jaccard_max']:.4f} | "
            f"zeros = {diag['jaccard_zero_pct']:.2f}%"
        ),
        (
            f"  Meta-grafo    : arestas = {diag['meta_edges']} | "
            f"grau = {diag['meta_deg_mean']:.2f} (min={diag['meta_deg_min']}, max={diag['meta_deg_max']}) | "
            f"comp. conexas = {diag['meta_components']}"
        ),
    ]
    if "pct_edges_train_unlabeled" in diag:
        split_msg = (
            f"  Acoplamento   : train-train = {diag['pct_edges_train_train']:.1f}% | "
            f"train-não_treino = {diag['pct_edges_train_unlabeled']:.1f}% | "
            f"não_treino-não_treino = {diag['pct_edges_unlabeled_unlabeled']:.1f}%"
        )
        if "pct_edges_train_val" in diag:
            split_msg += (
                f" (train-val={diag['pct_edges_train_val']:.1f}%, "
                f"train-test={diag['pct_edges_train_test']:.1f}%)"
            )
        lines.append(split_msg)
    if "homophily_pct" in diag:
        lines.append(
            f"  Homofilia (auditoria offline, sem uso no grafo): {diag['homophily_pct']:.2f}% arestas intra-classe"
        )
    return "\n".join(lines)


def build_visg2(
    X_total,
    k=3,
    nvg_mode="none",
    nvg_param=None,
    neighbor_policy="closest_band_union",
    meta_mode="unsupervised",
    y=None,
    train_mask=None,
    val_mask=None,
    test_mask=None,
    return_diagnostics=False,
    vg_cls=None,
):
    """
    Constrói o Meta-Grafo baseado no VisG2 (Lima Filho, Sabino-Silva & Carneiro, IJCNN 2025),
    parametrizado conforme a Árvore de Decisões do projeto:
      - Decisão 01 (NVG): nvg_mode ('none' | 'neighbor_limit' | 'band_window'), nvg_param (n ou w),
        e neighbor_policy ('closest_band_union' | 'right_n').
      - Decisão 02 (Meta-grafo): k e meta_mode ('unsupervised' | 'train_only' | 'train_supervised').
    """
    n_samples = len(X_total)

    # 1. Construção do NVG por espectro e extração determinística de arestas
    edge_sets = [
        _extract_nvg_edges(
            X_total[i],
            nvg_mode=nvg_mode,
            nvg_param=nvg_param,
            neighbor_policy=neighbor_policy,
            vg_cls=vg_cls,
        )
        for i in range(n_samples)
    ]

    # 2. Matriz de Similaridade de Jaccard Topológica (Eqs. 2–3)
    sim_matrix = _compute_jaccard_matrix(edge_sets)

    # 3. Formação do Meta-Grafo (Decisão 02)
    # Blindagem explícita: modos não-supervisionados jamais recebem o vetor y
    y_for_graph = y if meta_mode == "train_supervised" else None
    mask_for_graph = train_mask if meta_mode in ("train_only", "train_supervised") else None

    A = _build_topk_metagraph(
        sim_matrix,
        k=k,
        meta_mode=meta_mode,
        y=y_for_graph,
        train_mask=mask_for_graph,
    )

    if return_diagnostics:
        diag = compute_visg2_diagnostics(
            edge_sets=edge_sets,
            sim_matrix=sim_matrix,
            A=A,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return A, diag

    return A
