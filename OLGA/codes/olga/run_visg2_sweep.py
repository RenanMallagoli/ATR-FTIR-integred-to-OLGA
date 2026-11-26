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
        f1_cell = f"`{m['best_f1']}` *(ES ep. {m['es_epoch']})*" if m["es_epoch"] != "N/A" else f"`{m['best_f1']}`"
        lines.append(
            f"| **{exp_id}** | `{nvg_mode}` / `{policy}` | `{n}` | `{k}` | `{meta_mode}` | "
            f"`{m['nvg_edges']}` | `{m['jaccard']}` | `{m['meta_deg']}` | `{m['meta_comp']}` | "
            f"`{m['homophily']}` | {m['coupling']} | {f1_cell} | `{m['elapsed_s']}` |"
        )
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Orquestrador isolado da bateria VisG2 E01 + E01R")
    parser.add_argument("--dataset", type=str, default="ftir_diabetes", help="Nome do dataset para o CSV de saída")
    parser.add_argument("--dry-run", action="store_true", help="Apenas imprime os comandos sem executar")
    args = parser.parse_args()

    main_py = Path(__file__).resolve().parent / "main.py"
    total = len(E01_E01R_EXPERIMENTS)

    if args.dry_run:
        for idx, exp in enumerate(E01_E01R_EXPERIMENTS, start=1):
            cmd = build_command(main_py, args.dataset, exp)
            print(f"[{idx}/{total}] {' '.join(cmd)}")
        return

    results = []
    t_sweep = time.time()
    for idx, exp in enumerate(E01_E01R_EXPERIMENTS, start=1):
        exp_id, tag, nvg_mode, n, policy, k, meta_mode = exp
        cmd = build_command(main_py, args.dataset, exp)
        print(f"\n=== [{idx}/{total}] {exp_id} ({policy}, n={n}, k={k}, tag={tag}) ===", flush=True)

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

    print(f"\n=== Bateria E01 + E01R concluída em {time.time() - t_sweep:.1f}s ===\n", flush=True)
    print(format_markdown_summary(results), flush=True)


if __name__ == "__main__":
    main()
