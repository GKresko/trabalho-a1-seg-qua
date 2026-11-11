"""Gera um dataset de exemplo (vendas anuais de TVs) em CSV, JSON e XML."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # semente fixa: os dados saem iguais a cada execução

# cidade: (latitude, longitude, vendas base por loja em 2015)
cidades = {
    "Curitiba": (-25.43, -49.27, 800),
    "Florianopolis": (-27.59, -48.55, 500),
    "Porto Alegre": (-30.03, -51.22, 650),
    "Sao Paulo": (-23.55, -46.63, 1200),
    "Rio de Janeiro": (-22.91, -43.17, 1000),
}
marcas = {"Samsung": 0, "LG": -30, "Sony": 40}  # efeito de cada marca nas vendas
linhas = []
for cidade, (lat, lon, venda_base) in cidades.items():
    for ano in range(2015, 2025):
        for _ in range(10):  # 10 lojas por cidade/ano
            marca = rng.choice(list(marcas))
            tamanho = int(rng.choice([32, 43, 55, 65]))
            preco = rng.normal(1500 + tamanho * 25 - (ano - 2015) * 60, 150)  # preço sobe com o tamanho e cai com os anos
            # vendas crescem com o tempo, caem com o preço e têm ruído
            vendas = (venda_base * (1 + 0.06 * (ano - 2015)) - 0.1 * (preco - 2400)
                      + marcas[marca] + rng.normal(0, 40))
            linhas.append({"cidade": cidade, "latitude": lat, "longitude": lon,
                           "marca": marca, "tamanho": tamanho, "ano": ano,
                           "preco_medio": round(preco, 2), "vendas": int(vendas)})

df = pd.DataFrame(linhas)
df.to_csv("dados.csv", index=False)
df.to_json("dados.json", orient="records", force_ascii=False)
df.to_xml("dados.xml", index=False)
