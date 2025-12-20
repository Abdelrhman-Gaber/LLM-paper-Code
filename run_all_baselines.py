# run_all_baselines.py
import pandas as pd

from config import FLConfig
from data import load_framingham_dataset, simulate_clients, build_global_feature_space
from embedder import LLMEmbedder
from clients_llm import LLMClient
from clients_tabular import TabularClient
from algorithms_llm import federated_training_llm, evaluate_global_llm
from algorithms_baselines import (
    train_fedavg_tabular,
    train_fedprox,
    train_scaffold,
    train_clustered_fl,
    train_fedxgboost_central,
    train_mi_baseline,
    _evaluate_tabular_global as _eval_tab,
)
from plots import estimate_model_size_kb, plot_accuracy_vs_comm


def run_suite(csv_path: str, num_clients_list=(3, 5, 10)):
    results = []

    for num_clients in num_clients_list:
        print("=" * 80)
        print(f"### NUM CLIENTS = {num_clients} ###")

        df = load_framingham_dataset(csv_path)
        client_dfs = simulate_clients(df, num_clients=num_clients, imbalance=True)

        # --- FedLLM-Align (DistilBERT + NN, structured) ---
        config = FLConfig(num_clients=num_clients, backbone="distilbert", serialization_format="structured")
        embedder = LLMEmbedder(backbone=config.backbone, max_len=config.max_seq_len)

        llm_clients = [
            LLMClient(i, cdf, config.serialization_format, embedder) for i, cdf in enumerate(client_dfs)
        ]

        history_llm, model_llm = federated_training_llm(llm_clients, embedder, config)
        f1_llm, _ = evaluate_global_llm(llm_clients, model_llm)
        model_size_llm_kb = estimate_model_size_kb(model_llm)
        print(f"[FedLLM-Align] F1={f1_llm:.4f}, size~{model_size_llm_kb:.1f} KB")

        results.append(
            {
                "method": "FedLLM-Align (DistilBERT + NN)",
                "num_clients": num_clients,
                "f1": f1_llm,
                "model_size_kb": model_size_llm_kb,
            }
        )

        # --- Traditional baselines on tabular union-of-columns ---
        all_cols, X_clients, y_clients = build_global_feature_space(client_dfs)
        tab_clients = [TabularClient(i, Xc, yc) for i, (Xc, yc) in enumerate(zip(X_clients, y_clients))]
        input_dim = len(all_cols)

        # FedAvg (tabular)
        history_fedavg, model_fedavg = train_fedavg_tabular(tab_clients, input_dim, config)
        f1_fedavg, _ = _eval_tab(tab_clients, model_fedavg)
        size_fedavg_kb = estimate_model_size_kb(model_fedavg)
        results.append(
            {
                "method": "FedAvg (heterogeneous tabular)",
                "num_clients": num_clients,
                "f1": f1_fedavg,
                "model_size_kb": size_fedavg_kb,
            }
        )

        # FedProx
        history_prox, model_prox = train_fedprox(tab_clients, input_dim, config)
        f1_prox, _ = _eval_tab(tab_clients, model_prox)
        size_prox_kb = estimate_model_size_kb(model_prox)
        results.append(
            {
                "method": "FedProx",
                "num_clients": num_clients,
                "f1": f1_prox,
                "model_size_kb": size_prox_kb,
            }
        )

        # SCAFFOLD
        history_scaf, model_scaf = train_scaffold(tab_clients, input_dim, config)
        f1_scaf, _ = _eval_tab(tab_clients, model_scaf)
        size_scaf_kb = estimate_model_size_kb(model_scaf)
        results.append(
            {
                "method": "SCAFFOLD",
                "num_clients": num_clients,
                "f1": f1_scaf,
                "model_size_kb": size_scaf_kb,
            }
        )

        # Clustered FL
        history_cluster, model_cluster = train_clustered_fl(tab_clients, input_dim, config)
        f1_cluster, _ = _eval_tab(tab_clients, model_cluster)
        size_cluster_kb = estimate_model_size_kb(model_cluster)
        results.append(
            {
                "method": "Clustered FL",
                "num_clients": num_clients,
                "f1": f1_cluster,
                "model_size_kb": size_cluster_kb,
            }
        )

        # FedXGBoost (centralized approx)
        f1_fxgb, _, size_fxgb_kb = train_fedxgboost_central(X_clients, y_clients)
        results.append(
            {
                "method": "FedXGBoost (central approx)",
                "num_clients": num_clients,
                "f1": f1_fxgb,
                "model_size_kb": size_fxgb_kb,
            }
        )

        # MI baseline
        f1_mi, _, size_mi_kb = train_mi_baseline(X_clients, y_clients, top_k=10)
        results.append(
            {
                "method": "Mutual-Info baseline (approx)",
                "num_clients": num_clients,
                "f1": f1_mi,
                "model_size_kb": size_mi_kb,
            }
        )

    df_res = pd.DataFrame(results)
    df_res.to_csv("fedllm_align_all_results.csv", index=False)
    print(df_res)

    subset = df_res[df_res["num_clients"] == 3]
    points = {
        row["method"]: (row["f1"], row["model_size_kb"]) for _, row in subset.iterrows()
    }
    plot_accuracy_vs_comm(points)


if __name__ == "__main__":
    csv_path = "framingham.csv"
    run_suite(csv_path)
