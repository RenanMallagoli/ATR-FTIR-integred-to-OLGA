import torch
import networkx as nx
from olga import OLGA
from oneclass import one_class_loss, one_class_masking, One_Class_GNN_prediction, EarlyStopping
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn.models.autoencoder import GAE
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import pandas as pd
import random
import os
import sys
import traceback
from pathlib import Path
import time
from utils import save_values, write_results, init_metrics
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
import argparse
from sklearn.neighbors import kneighbors_graph
import graph_builders as gb
import data_loader as dl

def train_olga(g, hidden1, hidden2, patience, lr, r, multi_task, epochs):
    loss_ocl = 0
    recon_loss_unsup = 0
    embeddings, losses_ocl, losses_rec, accuracies, losses = [], [], [], [], []
    best_embeddings, best_radius, best_center = [], 0, []
    best_epoch = 0
    best_pre_mt_f1, best_pre_mt_ep = -1.0, -1
    best_post_mt_f1, best_post_mt_ep = -1.0, -1
    c = [0] * hidden2
    center = torch.Tensor(c)
    radius = torch.Tensor([r])


    mask, t_mask, mask_unsup, t_mask_unsup = one_class_masking(g, False)
    G = from_networkx(g)
    g_unsup = g.subgraph(t_mask_unsup)
    G_unsup = from_networkx(g_unsup)
    model_ocl = OLGA(len(G.features[0]), [hidden1, hidden2])
    model = GAE(model_ocl)
    stopper = EarlyStopping(patience)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mt_epoch = int(multi_task)

    for epoch in range(epochs+1):
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        learned_representations = model.encode(G.features.float(), G.edge_index)

        if epoch < multi_task:
            loss = model.recon_loss(learned_representations, G.edge_index)
        else:
            loss_ocl = one_class_loss(center, radius, learned_representations, mask)

            recon_loss_unsup = model.recon_loss(learned_representations[mask_unsup], G_unsup.edge_index)

            loss = loss_ocl + recon_loss_unsup

        f1 = One_Class_GNN_prediction(center, radius, learned_representations, g, 'val', True)['macro avg']['f1-score']

        if epoch < multi_task:
            if f1 > best_pre_mt_f1:
                best_pre_mt_f1, best_pre_mt_ep = f1, epoch
        else:
            if f1 > best_post_mt_f1:
                best_post_mt_f1, best_post_mt_ep = f1, epoch

        # Compute gradients
        loss.backward()

        # Tune parameters
        optimizer.step()

        stop, best_embeddings, best_radius, best_center, best_epoch = stopper.step(f1, loss, epoch, radius, center,
                                                                                   learned_representations)

        embeddings.append(learned_representations)
        losses_ocl.append(loss_ocl)
        losses_rec.append(recon_loss_unsup)
        losses.append(loss)
        accuracies.append(f1)

        if epoch % 100 == 0 or epoch in (mt_epoch - 1, mt_epoch):
            print(f"Época {epoch:04d} | F1-Score (Val): {f1:.4f} | Loss: {loss.item():.4f}", flush=True)
        # --------------------------------

        if stop:
            print(f"\n-> Treinamento interrompido (Early Stopping) na época {epoch}.", flush=True)
            print(f"-> Melhor F1-Score alcançado: {stopper.best_score:.4f}\n", flush=True)
            break

    dic_results = One_Class_GNN_prediction(best_center, best_radius, best_embeddings, g, 'val', True)
    dic_test = One_Class_GNN_prediction(best_center, best_radius, best_embeddings, g, 'test', True)

    print("[OLGA Diagnóstico da Ponte]", flush=True)
    print(
        f"  Transição ep.{mt_epoch} : Val Macro-F1 pré-150 = {best_pre_mt_f1:.4f} (ep. {best_pre_mt_ep}) "
        f"| pós-150 = {best_post_mt_f1:.4f} (ep. {best_post_mt_ep}) | best_epoch = {best_epoch}",
        flush=True,
    )
    print(
        f"  Validação (Val)  : Macro-F1 = {dic_results['macro avg']['f1-score']:.4f} "
        f"| F1(+1 int) = {dic_results['1']['f1-score']:.4f} "
        f"| F1(-1 out) = {dic_results['-1']['f1-score']:.4f} "
        f"| Acc = {dic_results['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"  Teste (Test)     : Macro-F1 = {dic_test['macro avg']['f1-score']:.4f} "
        f"| F1(+1 int) = {dic_test['1']['f1-score']:.4f} "
        f"| F1(-1 out) = {dic_test['-1']['f1-score']:.4f} "
        f"| Acc = {dic_test['accuracy']:.4f}",
        flush=True,
    )

    dic_results['_test_report'] = dic_test
    return dic_results

def train_parameters(l_graphs, patience, hidden1, hidden2, r, lr, file_name, pr, epochs):
    multi_task = patience / 2
    l_param = str(hidden1) + '_' + str(hidden2) + '_' + str(r) + '_' + str(lr) + '_' + str(patience)
    metrics = init_metrics()
    test_metrics = init_metrics()
    has_test_metrics = False
    for g in l_graphs:
        start = time.time()
        values = train_olga(g, hidden1, hidden2, patience, lr, r, multi_task,epochs)
        end = time.time()
        time_ = end - start
        test_values = values.pop('_test_report', None)
        values['time'] = time_
        save_values(metrics, values)
        if test_values is not None:
            test_values['time'] = time_
            save_values(test_metrics, test_values)
            has_test_metrics = True

    write_results(metrics, file_name, l_param, pr)
    if has_test_metrics:
        if file_name.endswith('_OLGA.csv'):
            test_file_name = file_name[:-len('_OLGA.csv')] + '_test_OLGA.csv'
        else:
            test_file_name = f"test_{file_name}"
        write_results(test_metrics, test_file_name, l_param, pr)

def build_oneclass_bridge_splits(y_total, interest_class="diabetes"):
    """
    Define as máscaras de partição (train/val/test) e o vetor de rótulos binários
    alinhado à convenção do oneclass.py (1 = classe de interesse, 0 = outlier).
    """
    if interest_class not in ("diabetes", "control"):
        raise ValueError(
            f"interest_class inválida: '{interest_class}'. Escolha entre ('diabetes', 'control')."
        )

    classes = np.unique(y_total)
    if len(classes) < 2:
        raise ValueError(f"Esperadas ao menos 2 classes em y_total, encontrado: {classes}")
    classe_controle = classes[0]  # 0 = Controle (Saudável)
    classe_anomalia = classes[1]  # 1 = Anomalia (Diabetes)

    control_idx = np.where(y_total == classe_controle)[0]
    anomaly_idx = np.where(y_total == classe_anomalia)[0]

    # Mantém a ordem exata dos shuffles do protocolo anterior para reprodutibilidade
    np.random.shuffle(control_idx)
    np.random.shuffle(anomaly_idx)

    if interest_class == "diabetes":
        interest_idx = anomaly_idx
        outlier_idx = control_idx
        labels_olga = (y_total == classe_anomalia).astype(int)
    else:
        interest_idx = control_idx
        outlier_idx = anomaly_idx
        labels_olga = (y_total == classe_controle).astype(int)

    num_nodes = len(y_total)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    n_train = int(len(interest_idx) * 0.70)
    n_val = int(len(interest_idx) * 0.15)
    n_out_val = len(outlier_idx) // 2

    train_mask[interest_idx[:n_train]] = True
    val_mask[interest_idx[n_train:n_train + n_val]] = True
    val_mask[outlier_idx[:n_out_val]] = True
    test_mask[interest_idx[n_train + n_val:]] = True
    test_mask[outlier_idx[n_out_val:]] = True

    return train_mask, val_mask, test_mask, labels_olga


def _load_job_env_defaults():
    """
    Lê parâmetros opcionais de experimento definidos no job.env da raiz do projeto
    (ou variáveis de ambiente), permitindo configurar rodadas no docker_lab sem editar código.
    Argumentos passados via CLI sempre têm precedência sobre estes defaults.
    """
    cfg = {}
    job_env_path = Path(__file__).resolve().parents[3] / "job.env"
    if job_env_path.is_file():
        with open(job_env_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                val = val.split("#", 1)[0].strip().strip('"').strip("'")
                cfg[key.strip()] = val

    for key in (
        "DATASET",
        "EXP_TAG",
        "GRAPH_MODEL",
        "GRAPH_K",
        "VISG2_NVG_MODE",
        "VISG2_N",
        "VISG2_NEIGHBOR_POLICY",
        "VISG2_META_MODE",
        "INTEREST_CLASS",
    ):
        if key in os.environ:
            cfg[key] = os.environ[key].strip()

    nvg_mode_default = cfg.get("VISG2_NVG_MODE", "neighbor_limit") or "neighbor_limit"
    raw_n = cfg.get("VISG2_N", "")
    if raw_n.lower() in ("none", "null"):
        visg2_n_default = None
    elif raw_n == "":
        visg2_n_default = None if nvg_mode_default == "none" else 8
    else:
        visg2_n_default = int(raw_n)

    return {
        "dataset": cfg.get("DATASET", "food") or "food",
        "k_tag": cfg.get("EXP_TAG", "k=1") or "k=1",
        "graph_model": cfg.get("GRAPH_MODEL", "knn") or "knn",
        "graph_k": int(cfg["GRAPH_K"]) if cfg.get("GRAPH_K") else 3,
        "visg2_nvg_mode": nvg_mode_default,
        "visg2_n": visg2_n_default,
        "visg2_neighbor_policy": cfg.get("VISG2_NEIGHBOR_POLICY", "closest_band_union") or "closest_band_union",
        "visg2_meta_mode": cfg.get("VISG2_META_MODE", "unsupervised") or "unsupervised",
        "interest_class": cfg.get("INTEREST_CLASS", "diabetes") or "diabetes",
    }


def _log_stage(msg, t0):
    elapsed = time.time() - t0
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        print(f"[+{elapsed:6.1f}s | RAM {rss_mb:7.1f} MB] {msg}", flush=True)
    except Exception:
        print(f"[+{elapsed:6.1f}s] {msg}", flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    t0 = time.time()
    try:
        env_defaults = _load_job_env_defaults()

        parser = argparse.ArgumentParser(description='OLGA')
        
        parser.add_argument("--k", type=str, default=env_defaults["k_tag"], help="Tag do experimento usada no nome do CSV de saída")
        
        parser.add_argument("--h1", type=int, default=48, help="neurons from first hidden layer")
        
        parser.add_argument("--h2", type=int, default=2, help="neurons from second hidden layer")
        
        parser.add_argument("--radius", type=float, default=0.5, help="hypershpere radius")
        
        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        
        parser.add_argument("--patience", type=int, default=300, help="patience for early stopping")
        
        parser.add_argument("--n-epochs", type=int, default=5000, help="training epochs")
        
        parser.add_argument("--dataset", type=str, default=env_defaults["dataset"], help="dataset")

        parser.add_argument("--graph_model", type=str, default=env_defaults["graph_model"], 
                            choices=["knn", "sknn", "mknn", "smknn", "cosine", "pearson", "dtw", "visg2"], 
                            help="Modelo de pré-processamento do grafo")

        parser.add_argument("--graph_k", type=int, default=env_defaults["graph_k"],
                            help="k efetivo para formação do meta-grafo VisG2 (padrão validado: 3)")

        parser.add_argument("--visg2_nvg_mode", type=str, default=env_defaults["visg2_nvg_mode"],
                            choices=["none", "neighbor_limit", "band_window"],
                            help="Decisão 01: restrição de esparsidade no NVG (padrão validado: neighbor_limit; 'none' reproduz E00)")

        parser.add_argument("--visg2_n", type=int, default=env_defaults["visg2_n"],
                            help="Decisão 01: parâmetro n (limite de vizinhos, padrão validado: 8) ou w (janela de bandas) no NVG")

        parser.add_argument("--visg2_neighbor_policy", type=str, default=env_defaults["visg2_neighbor_policy"],
                            choices=["closest_band_union", "right_n"],
                            help="Sub-decisão 01: política de seleção dos n vizinhos visíveis (padrão: closest_band_union)")

        parser.add_argument("--visg2_meta_mode", type=str, default=env_defaults["visg2_meta_mode"],
                            choices=["unsupervised", "train_only", "train_supervised"],
                            help="Decisão 02: critério de conexão no meta-grafo VisG2 (padrão baseline E00: unsupervised)")

        parser.add_argument("--interest_class", type=str, default=env_defaults["interest_class"],
                            choices=["diabetes", "control"],
                            help="Decisão 03: classe de interesse encapsulada no OLGA (padrão: diabetes)")
        
        args = parser.parse_args()

        # Compõe o nome único do CSV para o write_results(utils.py), que grava em OLGA/codes/results/<file_name>
        file_name = f"{args.dataset}_{args.k}_OLGA.csv"
        _log_stage(
            f"Início | dataset={args.dataset} | graph_model={args.graph_model} | saida={file_name}",
            t0,
        )

        l_graphs = []

        np.random.seed(42)
        wavenumbers = np.linspace(1800, 900, 500)
        base_peaks = [(0.8, 1650, 20), (0.5, 1550, 25), (0.3, 1050, 15)]

        X_normal, X_anomaly = [], []
        for _ in range(50):
            y = np.zeros_like(wavenumbers)
            for A, mu, sigma in base_peaks:
                y += (A + np.random.normal(0, 0.05)) * np.exp(-((wavenumbers - mu)**2) / (2 * sigma**2))
            X_normal.append(y + np.random.normal(0, 0.02, size=500))
            
        for _ in range(10):
            y = np.zeros_like(wavenumbers)
            for A, mu, sigma in base_peaks:
                y += (A * np.random.uniform(0.5, 1.5)) * np.exp(-((wavenumbers - (mu + np.random.normal(0, 5)))**2) / (2 * sigma**2))
            X_anomaly.append(y + np.random.normal(0, 0.02, size=500))
            
        script_dir = Path(__file__).resolve().parent
        train_path = Path("dataset/train/train.csv")
        if not train_path.is_file():
            train_path = script_dir / "dataset" / "train" / "train.csv"
        test_path = Path("dataset/test/test.csv")
        if not test_path.is_file():
            test_path = script_dir / "dataset" / "test" / "test.csv"
        
        _log_stage(f"Carregando e processando dados reais ({train_path}, {test_path})...", t0)
        X_total, y_total, wavenumbers = dl.load_and_clean_real_data(train_path, test_path)
        _log_stage(f"Dados processados: X_total={X_total.shape}, dtype={X_total.dtype}", t0)

        train_mask, val_mask, test_mask, labels_olga = build_oneclass_bridge_splits(
            y_total, interest_class=args.interest_class
        )
        _log_stage(
            f"[Ponte OLGA] interest_class={args.interest_class} | "
            f"train={int(train_mask.sum())} (label=1) | val={int(val_mask.sum())} | test={int(test_mask.sum())}",
            t0,
        )

        # Roteador de Modelos de Grafo
        _log_stage(f"Construindo grafo '{args.graph_model}'...", t0)
        if args.graph_model == "knn":
            A_knn = gb.build_knn(X_total, k=3)
        elif args.graph_model == "sknn":    
            A_knn = gb.build_sknn(X_total, k=3)
        elif args.graph_model == "mknn":
            A_knn = gb.build_mknn(X_total, k=3)
        elif args.graph_model == "smknn":
            A_knn = gb.build_smknn(X_total, k=3)
        elif args.graph_model == "cosine":
            A_knn = gb.build_cosine_knn(X_total, k=3)
        elif args.graph_model == "pearson":
            A_knn = gb.build_pearson_network(X_total, threshold=0.95)
        elif args.graph_model == "dtw":
            A_knn = gb.build_dtw_knn(X_total, k=3)
        elif args.graph_model == "visg2": 
            A_knn, visg2_diag = gb.build_visg2(
                X_total,
                k=args.graph_k,
                nvg_mode=args.visg2_nvg_mode,
                nvg_param=args.visg2_n,
                neighbor_policy=args.visg2_neighbor_policy,
                meta_mode=args.visg2_meta_mode,
                y=y_total,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                return_diagnostics=True,
            )
            print(
                f"[VisG2 Config] k={args.graph_k} | nvg_mode={args.visg2_nvg_mode} | "
                f"n={args.visg2_n} | policy={args.visg2_neighbor_policy} | meta_mode={args.visg2_meta_mode}",
                flush=True,
            )
            print(gb.format_visg2_diagnostics(visg2_diag), flush=True)
        _log_stage(f"Grafo '{args.graph_model}' construído (shape={A_knn.shape}).", t0)

        # Convertendo para PyTorch
        x_tensor = torch.FloatTensor(X_total)
        y_tensor = torch.LongTensor(y_total.copy())
        rows, cols = np.where(A_knn == 1)
        edge_index = torch.LongTensor(np.vstack((rows, cols)))

        # =====================================================================
        # 2. PONTE: ADAPTANDO PARA O FORMATO NETWORKX ESPERADO PELO OLGA
        # =====================================================================
        g = nx.from_numpy_array(A_knn)

        # Injetamos as features, rótulo clínico bruto ('y') e polaridade OLGA ('label': 1=interest, 0=outlier)
        for i in range(len(X_total)):
            g.nodes[i]['features'] = X_total[i].tolist()
            
            g.nodes[i]['y'] = int(y_total[i])
            g.nodes[i]['label'] = int(labels_olga[i]) 
            
            g.nodes[i]['train'] = 1 if train_mask[i] else 0
            g.nodes[i]['val'] = 1 if val_mask[i] else 0
            g.nodes[i]['test'] = 1 if test_mask[i] else 0

        l_graphs = [g]

        # seeds
        seed = 81
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)   
        torch.cuda.manual_seed_all(seed)
        
        pr = '../results/' + args.dataset + '_' + args.k + '_'
        
        _log_stage("Iniciando treinamento OLGA (train_parameters)...", t0)
        train_parameters(l_graphs, args.patience, args.h1, args.h2, args.radius, args.lr, file_name, pr, args.n_epochs)
        test_out_name = (
            file_name[:-len("_OLGA.csv")] + "_test_OLGA.csv"
            if file_name.endswith("_OLGA.csv")
            else f"test_{file_name}"
        )
        _log_stage(
            f"Execução concluída. Métricas gravadas em: OLGA/codes/results/{file_name} "
            f"e OLGA/codes/results/{test_out_name}",
            t0,
        )
    except Exception:
        print("\n[ERRO FATAL] Exceção não tratada durante a execução:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
