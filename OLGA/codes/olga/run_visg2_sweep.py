import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

E01_E01R_EXPERIMENTS = [
    ("E01_n3", "visg2_E01_union_n3", "neighbor_limit", 3, "closest_band_union", 3, "unsupervised"),
    ("E01_n4", "visg2_E01_union_n4", "neighbor_limit", 4, "closest_band_union", 3, "unsupervised"),
    ("E01_n6", "visg2_E01_union_n6", "neighbor_limit", 6, "closest_band_union", 3, "unsupervised"),
    ("E01_n8", "visg2_E01_union_n8", "neighbor_limit", 8, "closest_band_union", 3, "unsupervised"),
    ("E01_n10", "visg2_E01_union_n10", "neighbor_limit", 10, "closest_band_union", 3, "unsupervised"),
    ("E01R_n2", "visg2_E01R_right_n2", "neighbor_limit", 2, "right_n", 3, "unsupervised"),
    ("E01R_n3", "visg2_E01R_right_n3", "neighbor_limit", 3, "right_n", 3, "unsupervised"),
    ("E01R_n5", "visg2_E01R_right_n5", "neighbor_limit", 5, "right_n", 3, "unsupervised"),
]

SWEEP_BATTERIES = {
    "E01_E01R": E01_E01R_EXPERIMENTS,
    "E02_E04": [
        ("E02_k2", "visg2_E02_union_n8_k2", "neighbor_limit", 8, "closest_band_union", 2, "unsupervised"),
        ("E02_k5", "visg2_E02_union_n8_k5", "neighbor_limit", 8, "closest_band_union", 5, "unsupervised"),
        ("E02_k7", "visg2_E02_union_n8_k7", "neighbor_limit", 8, "closest_band_union", 7, "unsupervised"),
        ("E03_w5", "visg2_E03_win5", "band_window", 5, "closest_band_union", 3, "unsupervised"),
        ("E03_w10", "visg2_E03_win10", "band_window", 10, "closest_band_union", 3, "unsupervised"),
        ("E03_w25", "visg2_E03_win25", "band_window", 25, "closest_band_union", 3, "unsupervised"),
        ("E03_w50", "visg2_E03_win50", "band_window", 50, "closest_band_union", 3, "unsupervised"),
        ("E04_tsup", "visg2_E04_trainsup_n8_k3", "neighbor_limit", 8, "closest_band_union", 3, "train_supervised"),
    ],
}


def _extract(pattern, text, group=1, default="N/A"):
    match = re.search(pattern, text)
    return match.group(group) if match else default


def parse_run_metrics(output_text, elapsed_s):
    """Extrai métricas estruturais do VisG2 e resultado do OLGA a partir do stdout de main.py."""
    coupling_match = re.search(
        r"train-não_treino\s*=\s*([0-9.]+%)(?:.*?\(train-val=([0-9.]+%),\s*train-test=([0-9.]+%)\))?",
        output_text,
    )
    if coupling_match:
        coupling = coupling_match.group(1)
        if coupling_match.group(2) and coupling_match.group(3):
            coupling = f"{coupling} (`{coupling_match.group(2)}` val / `{coupling_match.group(3)}` test)"
    else:
        coupling = "N/A"

    return {
        "nvg_edges": _extract(r"NVG/espectro\s*:\s*arestas\s*=\s*([0-9.]+\s*±\s*[0-9.]+)", output_text),
        "jaccard": _extract(r"Jaccard\s*:\s*média\s*=\s*([0-9.]+\s*±\s*[0-9.]+)", output_text),
        "meta_deg": _extract(r"Meta-grafo\s*:.*?\bgrau\s*=\s*([0-9.]+)", output_text),
        "meta_comp": _extract(r"comp\.\s*conexas\s*=\s*(\d+)", output_text),
        "coupling": coupling,
        "homophily": _extract(r"Homofilia[^:]*:\s*([0-9.]+%)", output_text),
        "best_f1": _extract(r"Melhor F1-Score alcançado:\s*([0-9.]+)", output_text),
        "es_epoch": _extract(r"Early Stopping\)\s*na época\s*(\d+)", output_text),
        "elapsed_s": f"{elapsed_s:.1f}",
    }


def build_command(main_py, dataset, exp):
    """Monta a linha de comando CLI isolada para um experimento do main.py."""
    _, tag, nvg_mode, n, policy, k, meta_mode = exp
    return [
        sys.executable,
        str(main_py),
        "--dataset",
        dataset,
        "--graph_model",
        "visg2",
        "--graph_k",
        str(k),
        "--visg2_nvg_mode",
        nvg_mode,
        "--visg2_n",
        str(n),
        "--visg2_neighbor_policy",
        policy,
        "--visg2_meta_mode",
        meta_mode,
        "--k",
        tag,
    ]


def format_markdown_summary(rows):
    """Formata a tabela Markdown consolidada compatível com a Seção 5 do documento de auditoria."""
    header = (
        "| ID Exp. | `nvg_mode` / `policy` | `n` / `w` | `k` | `meta_mode` | "
        "Arestas médias/NVG | Jaccard médio ± dp | Grau médio meta | Comp. conexas | "
        "Homofilia (%) | % `train`–não-treino | Macro-F1 (Val) | Tempo (s) |\n"
        "| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    lines = [header]
    for exp, m in rows:
        exp_id, _, nvg_mode, n, policy, k, meta_mode = exp
        mode_cell = "`band_window` / `-`" if nvg_mode == "band_window" else f"`{nvg_mode}` / `{policy}`"
        f1_cell = f"`{m['best_f1']}` *(ES ep. {m['es_epoch']})*" if m["es_epoch"] != "N/A" else f"`{m['best_f1']}`"
        lines.append(
            f"| **{exp_id}** | {mode_cell} | `{n}` | `{k}` | `{meta_mode}` | "
            f"`{m['nvg_edges']}` | `{m['jaccard']}` | `{m['meta_deg']}` | `{m['meta_comp']}` | "
            f"`{m['homophily']}` | {m['coupling']} | {f1_cell} | `{m['elapsed_s']}` |"
        )
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Orquestrador isolado das baterias VisG2")
    parser.add_argument(
        "--battery",
        type=str,
        default="E02_E04",
        choices=list(SWEEP_BATTERIES.keys()),
        help="Bateria de experimentos a executar (padrão: E02_E04)",
    )
    parser.add_argument("--dataset", type=str, default="ftir_diabetes", help="Nome do dataset para o CSV de saída")
    parser.add_argument("--dry-run", action="store_true", help="Apenas imprime os comandos sem executar")
    args = parser.parse_args()

    experiments = SWEEP_BATTERIES[args.battery]
    main_py = Path(__file__).resolve().parent / "main.py"
    total = len(experiments)

    if args.dry_run:
        print(f"=== Dry-run bateria {args.battery} ({total} execuções) ===")
        for idx, exp in enumerate(experiments, start=1):
            cmd = build_command(main_py, args.dataset, exp)
            print(f"[{idx}/{total}] {' '.join(cmd)}")
        return

    print(f"=== Iniciando bateria {args.battery} ({total} execuções isoladas) ===", flush=True)
    results = []
    t_sweep = time.time()
    for idx, exp in enumerate(experiments, start=1):
        exp_id, tag, nvg_mode, n, policy, k, meta_mode = exp
        cmd = build_command(main_py, args.dataset, exp)
        print(
            f"\n=== [{idx}/{total}] [{args.battery}] {exp_id} "
            f"({nvg_mode}/{policy}, n={n}, k={k}, meta={meta_mode}, tag={tag}) ===",
            flush=True,
        )

        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        captured = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
        proc.wait()
        elapsed_s = time.time() - t0

        if proc.returncode != 0:
            print(f"\n[ERRO] Experimento {exp_id} falhou com código {proc.returncode}.", flush=True)
            sys.exit(proc.returncode)

        metrics = parse_run_metrics("".join(captured), elapsed_s)
        results.append((exp, metrics))

    print(f"\n=== Bateria {args.battery} concluída em {time.time() - t_sweep:.1f}s ===\n", flush=True)
    print(format_markdown_summary(results), flush=True)


if __name__ == "__main__":
    main()
