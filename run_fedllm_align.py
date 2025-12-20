# run_fedllm_align.py
from dataclasses import asdict

from config import FLConfig
from data import load_framingham_dataset, simulate_clients
from embedder import LLMEmbedder
from clients_llm import LLMClient
from algorithms_llm import federated_training_llm, evaluate_global_llm
from plots import plot_convergence, estimate_model_size_kb, plot_accuracy_vs_comm, plot_tsne_embeddings


def run_single_experiment(
    csv_path: str,
    num_clients: int = 3,
    backbone: str = "distilbert",
    serialization_format: str = "structured",
):
    config = FLConfig(
        num_clients=num_clients,
        backbone=backbone,
        serialization_format=serialization_format,
    )
    print("Config:", asdict(config))

    df = load_framingham_dataset(csv_path)
    client_dfs = simulate_clients(df, num_clients=num_clients, imbalance=True)

    embedder = LLMEmbedder(backbone=backbone, max_len=config.max_seq_len)

    clients = []
    for i, cdf in enumerate(client_dfs):
        clients.append(
            LLMClient(
                client_id=i,
                df=cdf,
                serialization_format=serialization_format,
                embedder=embedder,
            )
        )

    history, global_model = federated_training_llm(clients, embedder, config)
    f1_global, report = evaluate_global_llm(clients, global_model)
    print(f"[FedLLM-Align] Global F1: {f1_global:.4f}")

    plot_convergence(history, label=f"{backbone}-{num_clients}clients")
    model_size_kb = estimate_model_size_kb(global_model)
    plot_accuracy_vs_comm({f"{backbone}-{num_clients}": (f1_global, model_size_kb)})
    plot_tsne_embeddings(clients, embedder)

    return history, global_model, f1_global, report, model_size_kb


if __name__ == "__main__":
    csv_path = "framingham.csv"  # adjust path
    run_single_experiment(
        csv_path=csv_path,
        num_clients=3,
        backbone="distilbert",
        serialization_format="structured",
    )
