"""Gera um dataset de exemplo (vendas anuais de TVs) em CSV, JSON e XML.

Responsável: Giovanna (Dados e upload).
Os dados são SINTÉTICOS: não existe base pública com vendas de TVs por loja/cidade/ano,
então criamos dados com regras realistas (preço cai com os anos, vendas crescem etc.).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # semente fixa: os dados saem iguais a cada execução

# cidade: (latitude, longitude, vendas base por loja em 2015)
# latitude/longitude ficam na própria base para o mapa
cidades = {
    "Curitiba": (-25.43, -49.27, 800),
    "Florianopolis": (-27.59, -48.55, 500),
    "Porto Alegre": (-30.03, -51.22, 650),
    "Sao Paulo": (-23.55, -46.63, 1200),
    "Rio de Janeiro": (-22.91, -43.17, 1000),
    "Belo Horizonte": (-19.92, -43.94, 850),
    "Brasilia": (-15.79, -47.88, 750),
    "Salvador": (-12.98, -38.50, 700),
}
marcas = {"Samsung": 0, "LG": -30, "Sony": 40}  # efeito de cada marca nas vendas
linhas = []
# 8 cidades x 10 anos x 15 lojas = 1.200 registros (cada linha = uma loja em um ano)
for cidade, (lat, lon, venda_base) in cidades.items():
    for ano in range(2015, 2025):
        for _ in range(15):  # 15 lojas por cidade/ano
            marca = rng.choice(list(marcas))              # sorteia a marca
            tamanho = int(rng.choice([32, 43, 55, 65]))   # sorteia o tamanho (polegadas)
            # preço sobe com o tamanho e cai R$ 60 por ano (+ variação aleatória)
            preco = rng.normal(1500 + tamanho * 25 - (ano - 2015) * 60, 150)
            # vendas: crescem 6% ao ano, caem com o preço, mudam com a marca, + ruído aleatório
            vendas = (venda_base * (1 + 0.06 * (ano - 2015)) - 0.1 * (preco - 2400)
                      + marcas[marca] + rng.normal(0, 40))
            linhas.append({"cidade": cidade, "latitude": lat, "longitude": lon,
                           "marca": marca, "tamanho": tamanho, "ano": ano,
                           "preco_medio": round(preco, 2), "vendas": int(vendas)})

# Salva a mesma base nos 3 formatos aceitos pelo upload
df = pd.DataFrame(linhas)
df.to_csv("dados.csv", index=False)
df.to_json("dados.json", orient="records", force_ascii=False)
df.to_xml("dados.xml", index=False)
