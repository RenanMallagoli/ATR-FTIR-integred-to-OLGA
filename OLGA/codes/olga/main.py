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
import sys
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

        if epoch % 100 == 0:
            print(f"Época {epoch:04d} | F1-Score (Val): {f1:.4f} | Loss: {loss.item():.4f}")
        # --------------------------------

        if stop:
            print(f"\n-> Treinamento interrompido (Early Stopping) na época {epoch}.")
            print(f"-> Melhor F1-Score alcançado: {stopper.best_score:.4f}\n")
            break

    dic_results = One_Class_GNN_prediction(best_center, best_radius, best_embeddings, g, 'val', True)

    return dic_results

def train_parameters(l_graphs, patience, hidden1, hidden2, r, lr, file_name, pr, epochs):
    multi_task = patience / 2
    l_param = str(hidden1) + '_' + str(hidden2) + '_' + str(r) + '_' + str(lr) + '_' + str(patience)
    metrics = init_metrics()
    for g in l_graphs:
        start = time.time()
        values = train_olga(g, hidden1, hidden2, patience, lr, r, multi_task,epochs)
        end = time.time()
        time_ = end - start
        values['time'] = time_
        save_values(metrics, values)

    write_results(metrics, file_name, l_param, pr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OLGA')
    
    parser.add_argument("--k", type=str, default='k=1', help="k from graph modeling")
    
    parser.add_argument("--h1", type=int, default=48, help="neurons from first hidden layer")
    
    parser.add_argument("--h2", type=int, default=2, help="neurons from second hidden layer")
    
    parser.add_argument("--radius", type=float, default=0.5, help="hypershpere radius")
    
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    
    parser.add_argument("--patience", type=int, default=300, help="patience for early stopping")
    
    parser.add_argument("--n-epochs", type=int, default=5000, help="training epochs")
    
    parser.add_argument("--dataset", type=str, default="food", help="dataset")

    parser.add_argument("--graph_model", type=str, default="knn", 
                        choices=["knn", "sknn", "mknn", "smknn", "cosine", "pearson", "dtw", "visg2"], 
                        help="Modelo de pré-processamento do grafo")
    
    args = parser.parse_args()

    file_name = 'OLGA.csv' 

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
        
    train_path = "dataset/train/train.csv"
    test_path = "dataset/test/test.csv"
    
    print("Carregando e processando dados reais...")
    X_total, y_total, wavenumbers = dl.load_and_clean_real_data(train_path, test_path)

    # Roteador de Modelos de Grafo
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
        A_knn = gb.build_visg2(X_total, k=3)

    # Convertendo para PyTorch
    x_tensor = torch.FloatTensor(X_total)
    y_tensor = torch.LongTensor(y_total.copy())
    rows, cols = np.where(A_knn == 1)
    edge_index = torch.LongTensor(np.vstack((rows, cols)))

    num_nodes = len(y_total)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # 1. Identificando as classes automaticamente 
    # (Pega os dois valores únicos que existem no y_total, ex: 0 e 1)
    classes = np.unique(y_total)
    classe_controle = classes[0]  # Assumindo que o primeiro valor (ex: 0) é o Controle
    classe_anomalia = classes[1]  # O segundo valor (ex: 1) é o Diabetes

    # 2. Separando os índices de cada grupo
    control_idx = np.where(y_total == classe_controle)[0]
    anomaly_idx = np.where(y_total == classe_anomalia)[0]

    # Embaralha os índices para garantir que não haja viés na ordem do CSV
    np.random.shuffle(control_idx)
    np.random.shuffle(anomaly_idx)

    # 3. Definindo as proporções (70% treino, 15% validação, 15% teste para os controles)
    n_train = int(len(control_idx) * 0.70)
    n_val = int(len(control_idx) * 0.15)
    
    # TREINO: Recebe 70% dos Controles e ZERO Anomalias
    train_mask[control_idx[:n_train]] = True
    
    # VALIDAÇÃO: Recebe 15% dos Controles e METADE das Anomalias
    val_mask[control_idx[n_train:n_train+n_val]] = True
    val_mask[anomaly_idx[:len(anomaly_idx)//2]] = True
    
    # TESTE: Recebe os 15% restantes dos Controles e a outra METADE das Anomalias
    test_mask[control_idx[n_train+n_val:]] = True
    test_mask[anomaly_idx[len(anomaly_idx)//2:]] = True

    # =====================================================================
    # 2. PONTE: ADAPTANDO PARA O FORMATO NETWORKX ESPERADO PELO OLGA
    # =====================================================================
    g = nx.from_numpy_array(A_knn)

    # Injetamos as features, rótulos e as chaves exatas que o oneclass.py espera
    for i in range(len(X_total)):
        g.nodes[i]['features'] = X_total[i].tolist()
        
        # O modelo OLGA pode esperar que a classe de interesse seja 1 ou 0.
        # Caso o F1-score fique zerado ou ocorra erro de divisão, inverta os valores:
        # 1 para Controle, 0 para Anomalia. Aqui mantivemos 0 e 1, respectivamente.
        g.nodes[i]['y'] = int(y_total[i])
        g.nodes[i]['label'] = int(y_total[i]) 
        
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
    
    train_parameters(l_graphs, args.patience, args.h1, args.h2, args.radius, args.lr, file_name, pr, args.n_epochs)
