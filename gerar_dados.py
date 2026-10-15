"""Gera um dataset de exemplo (preço de imóveis por cidade)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # semente fixa: os dados saem iguais a cada execução
base = {"Curitiba": 5000, "Florianopolis": 8000, "Porto Alegre": 6000}  # preço do m² por cidade
linhas = []
for cidade, preco_m2 in base.items():
    n = 200
    area = rng.integers(30, 250, n)
    quartos = rng.integers(1, 5, n)
    # preço = área * valor do m² + adicional por quarto + ruído aleatório
    preco = area * preco_m2 + quartos * 20000 + rng.normal(0, 30000, n)
    linhas.append(pd.DataFrame({"cidade": cidade, "area": area,
                                "quartos": quartos, "preco": preco.round(2)}))
pd.concat(linhas).to_csv("dados.csv", index=False)
