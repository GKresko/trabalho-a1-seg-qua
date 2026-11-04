# Previsão de vendas de TVs por ano (ML com treino dinâmico)

App em Streamlit que analisa vendas de TVs por cidade e ano, treina um modelo de regressão e prevê vendas futuras.

## Funcionalidades
- Análise de dados (pandas) com limpeza e estatísticas descritivas
- Gráficos de barra e pizza (seaborn/matplotlib) e gráfico interativo (plotly)
- Escolha do modelo (Regressão Linear ou Random Forest) e ajuste de parâmetros
- **Upload / treino dinâmico**: envie um CSV com novos dados (ex.: nova cidade) na barra lateral,
  clique em "Incorporar dados" e depois em "Treinar / Re-treinar modelo"
- Predição a partir de cidade, ano e preço médio da TV

## Estrutura
- `app.py` – aplicação
- `gerar_dados.py` – gera `dados.csv` de exemplo
- `dados.csv` – base de dados (colunas: cidade, ano, preco_medio, vendas)

## Como rodar
```
pip install -r requirements.txt
python gerar_dados.py
streamlit run app.py
```

## Divisão de tarefas
Preencher com os nomes da equipe (ex.: dados, modelagem, interface, documentação).
