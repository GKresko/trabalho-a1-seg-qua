# Análise e previsão de vendas de TVs por ano

Aplicação web (Streamlit) para enviar uma base de dados, explorar os dados com gráficos e fazer previsões de vendas com machine learning.

## Funcionalidades
- **Upload** de arquivos `.csv`, `.json` ou `.xml` pela barra lateral. A base enviada substitui a atual
  (sem alterar o `dados.csv`) e o modelo treinado é descartado, para ser re-treinado com os dados novos.
  - Colunas obrigatórias: `cidade`, `ano`, `preco_medio`, `vendas`, `latitude`, `longitude`
  - Colunas extras (ex.: `marca`, `tamanho`) são usadas automaticamente no modelo
- **Filtros** de cidades e período, aplicados a todas as abas de análise
- **Análise de dados** com pandas: limpeza (`dropna`, `drop_duplicates`), resumo por cidade e estatísticas descritivas
- **Gráficos:** barra e pizza (matplotlib/seaborn); linha, dispersão e mapa interativos (plotly)
- **Modelos configuráveis:** Regressão Linear e Random Forest (nº de árvores e profundidade máxima),
  com métricas R² e MAE
- **Treino dinâmico:** o botão "Treinar / Re-treinar modelo" treina um modelo novo com a base atual
- **Predição** com formulário montado automaticamente a partir das colunas da base

## Abas da aplicação
| Aba | Conteúdo |
|---|---|
| Visão geral | Indicadores, resumo por cidade e estatísticas descritivas |
| Gráficos | Cada gráfico responde a uma pergunta: as vendas crescem? onde se vende mais? qual marca vende mais? o preço afeta as vendas? |
| Mapa | Total de vendas por cidade no mapa |
| Modelo e predição | Escolha do algoritmo, parâmetros, treino, métricas, gráficos real × previsto e importância das variáveis, e predição |
| Dados | Tabela com os dados filtrados |

Os filtros valem para as abas de análise; o modelo sempre treina com a base inteira.

## Estrutura
- `app.py` – aplicação
- `gerar_dados.py` – gera a base de exemplo (`dados.csv`, `dados.json`, `dados.xml`)
- `dados.*` – base de exemplo (dados sintéticos: 8 cidades, 2015–2024, 15 lojas por cidade/ano = 1.200 registros)

## Como rodar
```
pip install -r requirements.txt
python gerar_dados.py
streamlit run app.py
```
O app abre em http://localhost:8501.

## Divisão de tarefas
| # | Área | Integrante | Partes do código |
|---|---|---|---|
| 1 | Dados e upload | Giovanna | `gerar_dados.py`, leitura de CSV/JSON/XML, validação de colunas e tipos, limpeza |
| 2 | Interface e análise | Brian Cavalheiro | Configuração da página, filtros, abas, Visão geral e Dados |
| 3 | Visualização | Gustavo Claudino | Abas Gráficos e Mapa |
| 4 | Machine learning | Guilherme Tavares | Aba Modelo e predição (modelos, treino, métricas, gráficos do modelo e predição) |
