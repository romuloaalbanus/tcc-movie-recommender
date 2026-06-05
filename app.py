from flask import Flask, render_template, request, jsonify
from surprise import Dataset, KNNBasic, SVD, Reader
from surprise import accuracy
import pandas as pd
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-key")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
_poster_cache = {}  # filme_id -> poster_url ou None

# ── Carregar dataset e metadados dos filmes ──────────────────────────────────
print("Carregando MovieLens 100K...")
data = Dataset.load_builtin("ml-100k")
trainset_full = data.build_full_trainset()

# Mapear id interno -> nome do filme e gênero
raw_to_inner = trainset_full.to_inner_iid
inner_to_raw = trainset_full.to_raw_iid

item_file = "~/.surprise_data/ml-100k/ml-100k/u.item"
import os
item_path = os.path.expanduser(item_file)

filmes = {}
generos_cols = [
    "unknown","Action","Adventure","Animation","Children's","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
    "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

with open(item_path, encoding="latin-1") as f:
    for linha in f:
        partes = linha.strip().split("|")
        if len(partes) < 24:
            continue
        fid   = partes[0]
        titulo = partes[1]
        flags = [int(x) for x in partes[5:24]]
        genero = next((generos_cols[i] for i, v in enumerate(flags) if v == 1), "N/A")
        filmes[fid] = {"titulo": titulo, "genero": genero}

# 20 filmes populares para exibir na tela inicial (por número de avaliações)
contagem = {}
for uid, iid, _ in trainset_full.all_ratings():
    raw_iid = inner_to_raw(iid)
    contagem[raw_iid] = contagem.get(raw_iid, 0) + 1

filmes_populares = sorted(contagem, key=contagem.get, reverse=True)[:20]


def _buscar_poster(filme_id):
    if filme_id in _poster_cache:
        return
    titulo_raw = filmes.get(filme_id, {}).get("titulo", "")
    titulo = re.sub(r'\s*\(\d{4}\)$', '', titulo_raw).strip()
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie",
                         params={"api_key": TMDB_API_KEY, "query": titulo, "language": "en-US"},
                         timeout=4)
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            _poster_cache[filme_id] = "https://image.tmdb.org/t/p/w300" + results[0]["poster_path"]
            return
    except Exception:
        pass
    _poster_cache[filme_id] = None


def _precarregar_posters():
    import threading
    for fid in filmes_populares:
        threading.Thread(target=_buscar_poster, args=(fid,), daemon=True).start()


_precarregar_posters()


def treinar_e_recomendar(avaliacoes_usuario):
    """Treina os 3 modelos e retorna top-10 recomendações para cada um."""
    # Adicionar avaliações do usuário ao dataset
    novo_uid = "novo_usuario_9999"
    novas_linhas = [
        {"userID": novo_uid, "itemID": iid, "rating": nota}
        for iid, nota in avaliacoes_usuario.items()
    ]
    df_novo = pd.DataFrame(novas_linhas)

    # Montar dataset combinado
    df_original = pd.DataFrame(
        [(trainset_full.to_raw_uid(u), trainset_full.to_raw_iid(i), r)
         for u, i, r in trainset_full.all_ratings()],
        columns=["userID", "itemID", "rating"]
    )
    df_total = pd.concat([df_original, df_novo], ignore_index=True)

    reader = Reader(rating_scale=(1, 5))
    dataset_novo = Dataset.load_from_df(df_total[["userID", "itemID", "rating"]], reader)
    trainset_novo = dataset_novo.build_full_trainset()

    algoritmos = {
        "User-Based CF": KNNBasic(k=40, sim_options={"user_based": True,  "name": "pearson"}, verbose=False),
        "Item-Based CF": KNNBasic(k=40, sim_options={"user_based": False, "name": "pearson"}, verbose=False),
        "SVD":           SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02),
    }

    ja_avaliados = set(avaliacoes_usuario.keys())
    todos_itens  = set(filmes.keys())
    candidatos   = todos_itens - ja_avaliados

    recomendacoes = {}
    for nome, algo in algoritmos.items():
        algo.fit(trainset_novo)
        predicoes = [(iid, algo.predict(novo_uid, iid).est) for iid in candidatos]
        predicoes.sort(key=lambda x: x[1], reverse=True)
        top10 = [
            {
                "titulo": filmes[iid]["titulo"],
                "genero": filmes[iid]["genero"],
                "nota":   round(nota, 2),
            }
            for iid, nota in predicoes[:10]
        ]
        recomendacoes[nome] = top10

    return recomendacoes


@app.route("/poster/<filme_id>")
def poster(filme_id):
    if filme_id not in _poster_cache:
        _buscar_poster(filme_id)
    return jsonify(url=_poster_cache.get(filme_id))


@app.route("/", methods=["GET"])
def index():
    lista = [{"id": fid, **filmes[fid], "poster": _poster_cache.get(fid)} for fid in filmes_populares]
    return render_template("index.html", filmes=lista)


@app.route("/recomendar", methods=["POST"])
def recomendar():
    avaliacoes = {}
    for fid in filmes_populares:
        val = request.form.get(f"rating_{fid}", "").strip()
        if val:
            avaliacoes[fid] = float(val)

    if not avaliacoes:
        return render_template("index.html",
                               filmes=[{"id": fid, **filmes[fid]} for fid in filmes_populares],
                               erro="Avalie pelo menos um filme antes de gerar recomendações.")

    recomendacoes = treinar_e_recomendar(avaliacoes)
    return render_template("results.html", recomendacoes=recomendacoes)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(debug=debug, port=5001)
