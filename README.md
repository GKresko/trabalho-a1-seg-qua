# Previsão de preço de imóveis (ML com treino dinâmico)

App em Streamlit que analisa dados, treina um modelo de regressão e prevê preços.

## Funcionalidades
- Análise de dados (pandas) e gráficos (seaborn/matplotlib)
- Escolha do modelo (Regressão Linear ou Random Forest) e ajuste de parâmetros
- **Treino dinâmico**: envie um CSV com novos dados (ex.: nova cidade) na barra lateral,
  clique em "Incorporar dados" e depois em "Treinar / Re-treinar modelo"
- Predição a partir de cidade, área e quartos

## Estrutura
- `app.py` – aplicação
- `gerar_dados.py` – gera `dados.csv` de exemplo
- `dados.csv` – base de dados (colunas: cidade, area, quartos, preco)

## Como rodar
```
pip install -r requirements.txt
python gerar_dados.py
streamlit run app.py
```

## Divisão de tarefas
Preencher com os nomes da equipe (ex.: dados, modelagem, interface, documentação).
