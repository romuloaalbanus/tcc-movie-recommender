import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from surprise import Dataset, KNNBasic, SVD
from surprise.model_selection import cross_validate, KFold

os.makedirs("results", exist_ok=True)

# ── 1. Carregar dataset ──────────────────────────────────────────────────────
print("Carregando dataset MovieLens 100K...")
data = Dataset.load_builtin("ml-100k")

# ── 2. Instanciar algoritmos ─────────────────────────────────────────────────
algoritmos = {
    "User-Based CF": KNNBasic(k=40, sim_options={"user_based": True, "name": "pearson"}),
    "Item-Based CF": KNNBasic(k=40, sim_options={"user_based": False, "name": "pearson"}),
    "SVD":           SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02),
}

# ── 3. Validação cruzada — RMSE e MAE ───────────────────────────────────────
print("\nRodando validação cruzada (5 folds)...")
resultados_rmse_mae = {}

for nome, algo in algoritmos.items():
    print(f"  Avaliando {nome}...")
    cv = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=False)
    resultados_rmse_mae[nome] = {
        "RMSE": round(cv["test_rmse"].mean(), 2),
        "MAE":  round(cv["test_mae"].mean(), 2),
    }

df_rmse_mae = pd.DataFrame(resultados_rmse_mae).T
df_rmse_mae.index.name = "Algoritmo"
df_rmse_mae.to_csv("results/metricas_rmse_mae.csv")
print("\nRMSE e MAE:")
print(df_rmse_mae.to_string())

# ── 4. Precision@K e Recall@K ───────────────────────────────────────────────

def precision_recall_at_k(predictions, k, threshold=4.0):
    """Calcula Precision@K e Recall@K médios sobre todos os usuários."""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel         = sum(1 for _, true_r in user_ratings if true_r >= threshold)
        n_rec_k       = sum(1 for est, _     in top_k if est >= threshold)
        n_rel_and_rec = sum(1 for est, true_r in top_k if est >= threshold and true_r >= threshold)

        precisions.append(n_rec_k / k)
        recalls.append(n_rel_and_rec / n_rel if n_rel > 0 else 0)

    return round(np.mean(precisions), 2), round(np.mean(recalls), 2)


print("\nCalculando Precision@K e Recall@K...")
ks = [5, 10, 15, 20]
resultados_pk = {nome: {"K": [], "Precision": [], "Recall": []} for nome in algoritmos}

kf = KFold(n_splits=5)

for nome, algo in algoritmos.items():
    print(f"  Avaliando {nome}...")
    all_predictions = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        all_predictions += algo.test(testset)

    for k in ks:
        p, r = precision_recall_at_k(all_predictions, k)
        resultados_pk[nome]["K"].append(k)
        resultados_pk[nome]["Precision"].append(p)
        resultados_pk[nome]["Recall"].append(r)

rows = []
for nome, vals in resultados_pk.items():
    for i, k in enumerate(vals["K"]):
        rows.append({"Algoritmo": nome, "K": k,
                     "Precision": vals["Precision"][i],
                     "Recall":    vals["Recall"][i]})

df_pk = pd.DataFrame(rows)
df_pk.to_csv("results/metricas_precision_recall.csv", index=False)
print("\nPrecision@K e Recall@K:")
print(df_pk.to_string(index=False))

# ── 5. Gráficos ──────────────────────────────────────────────────────────────
cores = {"User-Based CF": "#2563eb", "Item-Based CF": "#16a34a", "SVD": "#dc2626"}

# Gráfico RMSE e MAE
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
metricas = ["RMSE", "MAE"]
nomes = list(resultados_rmse_mae.keys())

for ax, metrica in zip(axes, metricas):
    valores = [resultados_rmse_mae[n][metrica] for n in nomes]
    barras = ax.bar(nomes, valores, color=[cores[n] for n in nomes], width=0.5)
    ax.set_title(f"Comparação de {metrica}", fontsize=13, fontweight="bold")
    ax.set_ylabel(metrica)
    ax.set_ylim(0, max(valores) * 1.25)
    for barra, val in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.tick_params(axis="x", rotation=10)

fig.suptitle("RMSE e MAE — Validação Cruzada (5 folds)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/grafico_rmse_mae.png", dpi=150)
plt.close()

# Gráfico Precision@K
plt.figure(figsize=(8, 5))
for nome, vals in resultados_pk.items():
    plt.plot(vals["K"], vals["Precision"], marker="o", label=nome, color=cores[nome])
plt.title("Precision@K por Algoritmo", fontsize=13, fontweight="bold")
plt.xlabel("K")
plt.ylabel("Precision@K")
plt.xticks(ks)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/grafico_precision_k.png", dpi=150)
plt.close()

# Gráfico Recall@K
plt.figure(figsize=(8, 5))
for nome, vals in resultados_pk.items():
    plt.plot(vals["K"], vals["Recall"], marker="o", label=nome, color=cores[nome])
plt.title("Recall@K por Algoritmo", fontsize=13, fontweight="bold")
plt.xlabel("K")
plt.ylabel("Recall@K")
plt.xticks(ks)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/grafico_recall_k.png", dpi=150)
plt.close()

print("\n✓ Arquivos salvos em results/")
print("  - metricas_rmse_mae.csv")
print("  - metricas_precision_recall.csv")
print("  - grafico_rmse_mae.png")
print("  - grafico_precision_k.png")
print("  - grafico_recall_k.png")
