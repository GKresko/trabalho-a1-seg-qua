"""Gera um dataset de exemplo (vendas anuais de TVs por cidade)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # semente fixa: os dados saem iguais a cada execução
base = {"Curitiba": 800, "Florianopolis": 500, "Porto Alegre": 650}  # vendas base por loja em 2015
linhas = []
for cidade, venda_base in base.items():
    for ano in range(2015, 2025):
        for _ in range(10):  # 10 lojas por cidade/ano
            preco = rng.normal(2800 - (ano - 2015) * 80, 200)  # preço médio cai com os anos
            # vendas crescem com o tempo, caem quando o preço sobe, e têm ruído
            vendas = venda_base * (1 + 0.06 * (ano - 2015)) - 0.1 * (preco - 2400) + rng.normal(0, 40)
            linhas.append({"cidade": cidade, "ano": ano,
                           "preco_medio": round(preco, 2), "vendas": int(vendas)})
pd.DataFrame(linhas).to_csv("dados.csv", index=False)
