# Auditoria Técnica e Árvore de Decisões — Construção do VisG2 (ATR-FTIR + OLGA)

> **Projeto:** Iniciação Científica (UFU — Grupo AI Net) — Detecção não invasiva de diabetes via espectros ATR-FTIR com grafos e OLGA (*One-Class Graph Autoencoder*).  
> **Referência teórica base:** Lima Filho, Sabino-Silva & Carneiro (IJCNN 2025), *"High-level classification based on meta-graph of visibility graphs for Autism detection"*.  
> **Status desta etapa:** Plano estrutural validado e implementado localmente em `graph_builders.py` e `main.py`, pronto para sincronização e execução controlada no `docker-lab` remoto.

---

## 1. Escopo Estrito e Regras de Controle Experimental

- **Escopo ativo:** Auditar e evoluir **exclusivamente** a construção do meta-grafo **VisG2** (`build_visg2` em `OLGA/codes/olga/graph_builders.py`) e a passagem limpa de parâmetros via CLI no `OLGA/codes/olga/main.py`, mantendo os construtores de grafo desacoplados do modelo OLGA.
- **Fora de escopo (inalterado nesta etapa):**
  - Arquitetura, função de perda, mascaramento e rotina de treino/avaliação do modelo **OLGA** (`olga.py`, `oneclass.py`, `train_olga`, `train_parameters`).
  - Demais construtores de grafo (`build_knn`, `build_sknn`, `build_mknn`, `build_smknn`, `build_cosine_knn`, `build_pearson_network`, `build_dtw_knn`).
  - Fusão com metadados clínicos (`GENDER`, `AGE`, `HEMOGLOBIN`, `GLUCOSE`) e pipeline `ftir-prep` em `data_loader.py`.
- **Princípio de rastreabilidade:** Toda decisão que altera a topologia do NVG ou do meta-grafo está registrada abaixo como **Escolha Principal (ativa)** ou **Alternativa em Segundo Plano (comparativa)** e mapeada diretamente para uma flag CLI sem necessidade de editar código entre baterias de testes.

---

## 2. Diagnóstico da Auditoria do Código Base (Baseline `E00`)

### 2.1 Dados e Pré-processamento (`OLGA/codes/olga/data_loader.py`)
- **Entrada (`dataset/train/train.csv` e `dataset/test/test.csv`):**
  - `train.csv`: **520 amostras** rotuladas (`class == 0`: 250; `class == 1`: 270), com 6 colunas de metadados (`id`, `GENDER`, `AGE`, `HEMOGLOBIN`, `GLUCOSE`, `class`) e **3.736 bandas espectrais** estritamente crescentes de `399.1927` a `4000.605` $\text{cm}^{-1}$, com passo quase uniforme ($\Delta\tilde{\nu} \in [0.9639,\; 0.9650]\;\text{cm}^{-1}$).
  - `test.csv`: **312 amostras** sem a coluna `class` (removidas por `dropna(subset=["class"])`), restando as **520 amostras rotuladas**.
- **Pré-processamento (`ftir-prep==0.3.5`):**
  - Truncamento na região `'fingerprint'` ($\approx 933$ bandas em $900\text{–}1800\;\text{cm}^{-1}$), baseline `'rubberband'`, Savitzky-Golay (`polyorder=2`) e normalização `'minmax'`.
- **Implicação para o NVG:** Como o espaçamento espectral é praticamente constante ($\approx 0{,}964\;\text{cm}^{-1}$ por ponto) e a condição de visibilidade (Eq. 1) é invariante a transformações afins no eixo horizontal, a distância em índices $|b - a|$ traduz-se diretamente em distância espectral física ($\Delta\tilde{\nu} \approx 0{,}964 \cdot |b - a|\;\text{cm}^{-1}$).

### 2.2 Diagnóstico do `build_visg2` Original (`OLGA/codes/olga/graph_builders.py`)
- **Dependência:** `ts2vg==1.2.4` (`OLGA/codes/olga/requirements.txt`).
- **NVG por espectro:** Chamava `NaturalVG().build(X_total[i])` sem corte de vizinhança/alcance ($n$). Em espectros de $\sim 933$ pontos, picos dominantes atuam como *hubs* globais conectando centenas de bandas distantes e dominando o coeficiente de Jaccard.
  - *Nota sobre `ts2vg==1.2.4`:* `NaturalVG` não possui parâmetro nativo de limite de $n$ conexões por nó (`penetrable_limit` faz o inverso, aumentando a densidade; `weighted="abs_h_distance"` com `max_weight` usa desigualdade estrita `< max_weight` e muda `vg.edges` para triplas `(u, v, w)`). Por isso, a poda determinística pós-`NaturalVG` sobre pares canônicos `(min(u, v), max(u, v))` é a solução mais limpa, exata e auditável.
- **Matriz de Jaccard:** Fiel às Eqs. 2–3 ($|E_i \cap E_j| / |E_i \cup E_j|$) sobre o triângulo superior espelhado. Otimizada algebricamente via $|E_i \cup E_j| = |E_i| + |E_j| - |E_i \cap E_j|$ (resultado idêntico sem alocar $134\,940$ conjuntos temporários).
- **Meta-grafo:** Top-$k$ não-supervisionado simetrizado com `np.maximum(A, A.T)`. Adicionadas validação `1 <= k < n_samples` e ordenação estável (`kind="stable"`) para determinismo em empates de Jaccard.

### 2.3 Diagnóstico de Orquestração (`OLGA/codes/olga/main.py`)
- **`--k` legado vs. `k` real do grafo:** `--k` (`str`, default `'k=1'`) era usado apenas para nomear o CSV de saída (`pr`), enquanto o roteador fixava `k=3`. Adicionado `--graph_k` (`int`, default `3`) para controlar o $k$ real sem quebrar o prefixo `--k`.
- **Ordem entre máscaras de split e construção do grafo:** A geração de `train_mask`, `val_mask` e `test_mask` foi posicionada imediatamente antes do roteador de grafos, preservando exatamente o mesmo estado do gerador pseudoaleatório (`np.random.seed(42)`) do baseline `E00`. Isso permite:
  1. Auditar em modo somente-leitura o acoplamento estrutural entre `train`, `val` e `test` na Escolha Principal;
  2. Habilitar de forma limpa e sem *data leakage* as variantes restritas a `train_mask` na Decisão 02.
- **Observação lateral registrada (fora do escopo — código do OLGA intocado):** em `main.py`, `train_mask` usa `classe_controle = 0` (`label = 0`), enquanto `oneclass.py` (linhas 52–55) trata `label == 1` como `interest` e `label == 0` como `outlier`. Mantido inalterado conforme restrição estrita de escopo para verificação futura dedicada.

---

## 3. Árvore de Decisões do VisG2

```mermaid
flowchart TD
    V["Construção do VisG2 (build_visg2)"] --> D1["Decisão 01: Esparsidade no NVG (--visg2_nvg_mode, --visg2_n)"]
    V --> D2["Decisão 02: Conexão no Meta-Grafo (--graph_k, --visg2_meta_mode)"]

    D1 --> D10["BASELINE (E00): none (NVG irrestrito)"]
    D1 --> D1A["PRINCIPAL (Ativa): neighbor_limit (limite de n vizinhos visíveis por nó)"]
    D1 --> D1B["ALTERNATIVA (Posterior): band_window (janela máxima |b - a| <= w)"]

    D1A --> P1["Sub-decisão 01.1 (Padrão): closest_band_union (n mais próximos em |b - a|, união)"]
    D1A --> P2["Sub-decisão 01.2 (Comparativa): right_n (primeiros n visíveis à direita b > a)"]

    D2 --> D2A["PRINCIPAL (Ativa): unsupervised (top-k Jaccard sem rótulos + auditoria de acoplamento)"]
    D2 --> D2B1["ALTERNATIVA 02.A (One-Class Puro): train_only (candidatos restritos a train_mask)"]
    D2 --> D2B2["ALTERNATIVA 02.B (Eq. 4 Restrita ao Treino): train_supervised (y_i = y_j apenas em train_mask)"]
```

---

### Decisão 01 — Restrição de Vizinhança / Esparsidade no Grafo de Visibilidade (parâmetro $n$)

- **Motivação:** Controlar a densidade dos NVGs individuais para impedir que poucos picos globais dominem o conjunto de arestas $E_i$ e mascararem variações locais da assinatura bioquímica no coeficiente de Jaccard (alinhado ao uso de $n$ baixo no artigo IJCNN 2025).
- **Flags CLI correspondentes:** `--visg2_nvg_mode {none,neighbor_limit,band_window}` e `--visg2_n <int>`.

#### Escolha Principal (Ativa para Experimentação) — `nvg_mode="neighbor_limit"`
Limita o número de conexões visíveis mantidas por nó no NVG a partir de duas políticas determinísticas configuráveis via `--visg2_neighbor_policy`:

1. **Sub-decisão 01.1 (Padrão Inicial Validado) — `neighbor_policy="closest_band_union"`:**
   - Para cada nó $a \in \{0, \dots, L-1\}$, ordenam-se todos os seus vizinhos visíveis $b \in N_{\text{NVG}}(a)$ pela menor distância espectral $|b - a|$ (desempate determinístico pelo menor índice $b$) e selecionam-se os $n$ primeiros.
   - A aresta não-direcionada $(u, v)$ é mantida pela **união** das seleções ($v \in \text{top-}n(u) \lor u \in \text{top-}n(v)$), preservando simetria bilateral no espectro (bandas à esquerda e à direita de um pico/vale local).
   - **Propriedade matemática importante (limiar mínimo de $n$):** Em qualquer NVG 1D, pontos adjacentes $(a, a+1)$ são sempre visíveis entre si ($|b - a| = 1$). Logo, todo ponto interno $1 \le a \le L-2$ já consome **2 vagas** em vizinhos triviais imediatos $(a-1)$ e $(a+1)$ (comuns a 100% dos espectros). Assim, em `closest_band_union`, atalhos de visibilidade não-triviais ($|b - a| \ge 2$, que carregam a informação discriminativa real) passam a existir nos nós internos a partir de **$n \ge 3$** (cada nó interno retém até $n - 2$ atalhos não-triviais).
2. **Sub-decisão 01.2 (Alternativa de Política para Comparação) — `neighbor_policy="right_n"`:**
   - Seguindo a orientação sequencial $a < b$ da Eq. 1, cada ponto $a$ mantém apenas os seus primeiros $n$ vizinhos visíveis à direita ($b > a$, menores valores de $b - a$).
   - Como há apenas **1** vizinho trivial imediato à direita ($b = a + 1$), atalhos de visibilidade não-triviais já surgem a partir de **$n \ge 2$** (cada ponto $a$ retém até $n - 1$ atalhos à direita), com teto estrito $|E_i| \le n \cdot (L - 1)$.
   - *Equivalência de orçamento de atalhos por nó:* comparar `closest_band_union` com $n = m + 2$ frente a `right_n` com $n = m + 1$ ($m \ge 1$).

#### Alternativa em Segundo Plano (Para Teste Posterior) — `nvg_mode="band_window"`
- Mantém no NVG apenas arestas visíveis cuja distância em índices satisfaça $|b - a| \le w$ (horizonte espectral fixo $\Delta\tilde{\nu} \approx 0{,}964 \cdot w\;\text{cm}^{-1}$), avaliando proximidade morfológica estrita entre bandas adjacentes.

---

### Decisão 02 — Critério de Conexão no Meta-Grafo (Não-Supervisionado vs. Restrito ao Treino)

- **Motivação:** Conciliar a formulação do meta-grafo do VisG2 (Eq. 4 do artigo IJCNN 2025) com o aprendizado **one-class** transdutivo do OLGA sem introduzir *label leakage*.
- **Flags CLI correspondentes:** `--graph_k <int>` e `--visg2_meta_mode {unsupervised,train_only,train_supervised}`.

#### Escolha Principal (Ativa Agora) — `meta_mode="unsupervised"`
- Conecta cada super-nó aos seus $k$ vizinhos de maior similaridade de Jaccard em `X_total` (excluindo auto-laços), simetrizado via `np.maximum(A, A.T)`, **sem acessar `y_total`**.
- **Monitoramento automático em cada execução:** `compute_visg2_diagnostics` imprime no console:
  - Densidade dos NVGs (média, desvio, mín, máx de arestas por espectro);
  - Estatísticas da matriz de Jaccard (média, desvio, mediana, máx, % de pares com Jaccard zero);
  - Topologia do meta-grafo (arestas totais, grau médio/mín/máx, nº de componentes conexas);
  - **Acoplamento entre partições:** % de arestas `train–train`, `train–val`, `train–test` e `val/test–val/test`;
  - **Homofilia diagnóstica (somente leitura pós-construção):** % de arestas onde $y_i = y_j$, permitindo verificar se o ajuste de $n$ reduz conexões espúrias entre controles e diabéticos antes mesmo do treino da GNN.

#### Alternativas em Segundo Plano (Para Teste Posterior Sem Vazamento de Rótulos)
1. **`meta_mode="train_only"` (Controle de acoplamento transdutivo no regime one-class):**
   - Restringe os candidatos top-$k$ apenas aos nós pertencentes a `train_mask` (sem usar rótulos de `val`/`test`). Após simetrização `np.maximum(A, A.T)`, nós de `train` conectam-se entre si sem interferência de `val`/`test`, e nós de `val`/`test` ancoram-se aos seus $k$ controles de treino mais similares.
2. **`meta_mode="train_supervised"` (Eq. 4 restrita estritamente ao conjunto de treino):**
   - Para nós em `train_mask`, exige $y_i = y_j$ entre candidatos de `train_mask` (os rótulos fora de `train_mask` nunca são lidos). Nós fora de `train_mask` conectam-se aos seus top-$k$ em `train_mask` sem consulta de rótulo.
   - *Nota matemática:* Como o `train_mask` atual do OLGA contém apenas uma classe (`classe_controle`), $y_i = y_j$ já vale para todo par em `train_mask` (tornando `train_supervised` estruturalmente equivalente a `train_only` no one-class puro, mas já pronto caso um protocolo futuro tenha múltiplas classes rotuladas no treino).

---

## 4. Guia Rápido de Execução e Substituição no `docker-lab`

Todos os padrões CLI foram mantidos idênticos ao baseline `E00` (`--graph_k 3 --visg2_nvg_mode none --visg2_meta_mode unsupervised`), garantindo retrocompatibilidade total. Use `--k <tag>` para identificar cada configuração no arquivo de resultados (`../results/<dataset>_<tag>_OLGA.csv`).

### Comandos Prontos por Etapa Experimental (executar em `OLGA/codes/olga/`)

```bash
# E00 — Baseline atual (NVG irrestrito, k=3, meta-grafo não-supervisionado)
docker-lab run main.py --graph_model visg2 --k visg2_E00_base

# E01 — Decisão 01 Principal (neighbor_limit com política padrão closest_band_union; usar n >= 3)
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 3 --visg2_neighbor_policy closest_band_union --k visg2_E01_union_n3
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 4 --visg2_neighbor_policy closest_band_union --k visg2_E01_union_n4
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 6 --visg2_neighbor_policy closest_band_union --k visg2_E01_union_n6
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 8 --visg2_neighbor_policy closest_band_union --k visg2_E01_union_n8
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 10 --visg2_neighbor_policy closest_band_union --k visg2_E01_union_n10

# E01R — Comparação com Sub-decisão 01.2 (política right_n; n=2 e n=4 equivalem a 1 e 3 atalhos à direita)
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 2 --visg2_neighbor_policy right_n --k visg2_E01R_right_n2
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 3 --visg2_neighbor_policy right_n --k visg2_E01R_right_n3
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 5 --visg2_neighbor_policy right_n --k visg2_E01R_right_n5

# E02 — Sensibilidade ao k do meta-grafo (fixando o melhor n da etapa E01, ex.: n=5)
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 5 --graph_k 5 --k visg2_E02_n5_k5
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 5 --graph_k 7 --k visg2_E02_n5_k7

# E03 — Decisão 01 Alternativa em segundo plano (band_window: w em índices de banda)
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode band_window --visg2_n 10 --k visg2_E03_win10
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode band_window --visg2_n 25 --k visg2_E03_win25

# E04 — Decisão 02 Alternativa em segundo plano (restrição a train_mask sem label leakage)
docker-lab run main.py --graph_model visg2 --visg2_nvg_mode neighbor_limit --visg2_n 5 --visg2_meta_mode train_supervised --k visg2_E04_trainsup_n5
```

---

## 5. Registro de Resultados Experimentais (`docker-lab`)

| ID Exp. | `nvg_mode` / `policy` | `n` / `w` | `k` | `meta_mode` | Arestas médias/NVG | Jaccard médio ± dp | Grau médio meta | Comp. conexas | Homofilia (%) | % `train`–não-treino | Macro-F1 (Val) | Tempo (s) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E00** | `none` / `-` | `None` | `3` | `unsupervised` | _pendente_ | _pendente_ | _pendente_ | _pendente_ | _pendente_ | _pendente_ | _pendente_ | _pendente_ |
