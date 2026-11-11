# Análise e previsão de vendas de TVs por ano

Aplicação web (Streamlit) para enviar uma base de dados, visualizar gráficos e fazer previsões com machine learning.

## Funcionalidades
- **Upload** de arquivos `.csv`, `.json` ou `.xml` (colunas obrigatórias: `cidade`, `latitude`, `longitude`, `vendas`;
  as demais colunas, como `marca`, `tamanho`, `ano` e `preco_medio`, são usadas automaticamente no modelo)
- **Análise de dados** com pandas (limpeza e estatísticas descritivas)
- **Gráficos:** barra e pizza (matplotlib/seaborn), mapa de vendas por cidade, linha e dispersão interativos (plotly)
- **Modelos configuráveis:** Regressão Linear, Árvore de Decisão (profundidade), KNN (nº de vizinhos) e Random Forest (árvores e profundidade)
- **Treino dinâmico:** ao enviar uma nova base, basta clicar em "Treinar / Re-treinar modelo"
- **Predição** com formulário montado automaticamente a partir das colunas da base

## Estrutura
- `app.py` – aplicação
- `gerar_dados.py` – gera a base de exemplo (`dados.csv`, `dados.json`, `dados.xml`; dados sintéticos)

## Como rodar
```
pip install -r requirements.txt
python gerar_dados.py
streamlit run app.py
```

## Divisão de tarefas
Preencher com os nomes da equipe.
