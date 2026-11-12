"""App Streamlit: upload de dados, análise visual e ML com treino dinâmico (vendas de TVs).

Como o Streamlit funciona: a cada clique, o script roda inteiro de novo, de cima para baixo.
O que precisa sobreviver entre cliques (modelo treinado, métricas) fica no st.session_state.

Divisão do código (cada pessoa defende o seu bloco):
  1. Dados e upload ........ Giovanna           (ler_arquivo, upload, validação, limpeza)
  2. Interface e análise ... Brian Cavalheiro   (página, filtros, abas, Visão geral, Dados)
  3. Visualização .......... Gustavo Claudino   (abas Gráficos e Mapa)
  4. Machine learning ...... Guilherme Tavares  (aba Modelo e predição)
"""
import matplotlib.pyplot as plt   # gráficos estáticos (barra e pizza)
import pandas as pd               # leitura, limpeza e agregação dos dados
import plotly.express as px       # gráficos interativos (zoom, hover, mapa)
import seaborn as sns             # gráfico de barras sobre o matplotlib
import streamlit as st            # interface web
from sklearn.compose import ColumnTransformer          # aplica transformações por coluna
from sklearn.ensemble import RandomForestRegressor     # modelo 2: várias árvores votando
from sklearn.linear_model import LinearRegression      # modelo 1: reta de tendência
from sklearn.metrics import mean_absolute_error, r2_score  # métricas de avaliação
from sklearn.model_selection import train_test_split   # divide treino/teste
from sklearn.pipeline import Pipeline                  # junta pré-processamento + modelo
from sklearn.preprocessing import OneHotEncoder        # transforma texto em colunas 0/1

ALVO = "vendas"                       # o que o modelo prevê
GEO = ["latitude", "longitude"]       # só para o mapa (não entram no modelo)
# Sem estas colunas os gráficos e filtros não funcionam; extras (marca, tamanho) são opcionais
OBRIGATORIAS = ["cidade", "ano", "preco_medio", ALVO] + GEO

# =====================================================================================
# 2. INTERFACE (Brian) — configuração da página
# =====================================================================================
st.set_page_config(page_title="Vendas de TVs", layout="wide")  # layout largo: usa a tela toda
st.title("Análise e previsão de vendas de TVs por ano")


# =====================================================================================
# 1. DADOS E UPLOAD (Giovanna)
# =====================================================================================
def ler_arquivo(arquivo):
    """Lê CSV, JSON ou XML conforme a extensão do arquivo enviado."""
    nome = arquivo.name.lower()
    if nome.endswith(".csv"):
        return pd.read_csv(arquivo)
    if nome.endswith(".json"):
        return pd.read_json(arquivo)   # espera uma lista de registros
    if nome.endswith(".xml"):
        return pd.read_xml(arquivo)    # usa a biblioteca lxml (está no requirements)
    raise ValueError("Formato não suportado (use .csv, .xml ou .json)")


# Upload na barra lateral: a base enviada SUBSTITUI a de exemplo (não grava no disco)
st.sidebar.header("Base de dados")
arquivo = st.sidebar.file_uploader("Enviar base (CSV, JSON ou XML)", type=["csv", "json", "xml"])
if arquivo is not None:
    # Validação 1: o arquivo precisa ser legível (try/except evita que o app quebre)
    try:
        base = ler_arquivo(arquivo)
    except Exception as erro:
        st.sidebar.error(f"Erro ao ler o arquivo: {erro}")
        st.stop()  # interrompe o script aqui: nada abaixo é executado
    # Validação 2: precisa ter todas as colunas obrigatórias
    faltando = [c for c in OBRIGATORIAS if c not in base.columns]
    if faltando:
        st.sidebar.error(f"Colunas obrigatórias ausentes: {faltando}")
        st.stop()
    # Validação 3: colunas usadas em contas precisam ser números (ex.: "773 un" é rejeitado)
    nao_numericas = [c for c in ["ano", "preco_medio", ALVO] + GEO
                     if not pd.api.types.is_numeric_dtype(base[c])]
    if nao_numericas:
        st.sidebar.error(f"Estas colunas devem conter apenas números: {nao_numericas}")
        st.stop()
    st.sidebar.success(f"{arquivo.name} carregado")
else:
    base = pd.read_csv("dados.csv")  # sem upload: usa a base de exemplo (gerar_dados.py)
    st.sidebar.info("Usando a base de exemplo (dados.csv)")

# Treino dinâmico: se a base mudou, o modelo antigo é descartado (precisa re-treinar).
# A "assinatura" identifica a base pelo nome do arquivo + nº de linhas.
assinatura = (arquivo.name if arquivo else "exemplo", len(base))
if st.session_state.get("assinatura") != assinatura:
    st.session_state.assinatura = assinatura
    st.session_state.pop("modelo", None)    # pop(..., None): remove se existir
    st.session_state.pop("metricas", None)
    st.session_state.pop("teste", None)

# Limpeza: remove linhas com valores faltantes (dropna) e linhas repetidas (drop_duplicates)
base = base.dropna().drop_duplicates()

# =====================================================================================
# 2. INTERFACE E ANÁLISE (Brian) — filtros, abas, Visão geral e Dados
# =====================================================================================
# Filtros: afetam só a análise. O modelo treina com a base inteira (mais dados = melhor)
st.sidebar.header("Filtros")
todas_cidades = sorted(base["cidade"].unique())
cidades = st.sidebar.multiselect("Cidades", todas_cidades, default=todas_cidades)
ano_min, ano_max = int(base["ano"].min()), int(base["ano"].max())
if ano_min < ano_max:
    anos = st.sidebar.slider("Período", ano_min, ano_max, (ano_min, ano_max))  # devolve (início, fim)
else:  # base com um único ano: o slider não aceita início igual ao fim
    anos = (ano_min, ano_max)

# df = base filtrada; é ela que alimenta todas as abas de análise
df = base[base["cidade"].isin(cidades) & base["ano"].between(*anos)]
if df.empty:
    st.warning("Nenhum dado para os filtros escolhidos.")
    st.stop()

# Total por cidade, calculado uma vez e reaproveitado na barra e no mapa
por_cidade = df.groupby("cidade", as_index=False).agg(
    vendas=(ALVO, "sum"), latitude=("latitude", "mean"), longitude=("longitude", "mean"))

# Abas: o que estiver dentro de "with aba_x:" aparece naquela aba
aba_geral, aba_graficos, aba_mapa, aba_modelo, aba_dados = st.tabs(
    ["Visão geral", "Gráficos", "Mapa", "Modelo e predição", "Dados"])

# ---------- Aba: Visão geral (Brian) ----------
with aba_geral:
    # Cartões com os números principais
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", len(df))
    c2.metric("Cidades", df["cidade"].nunique())
    c3.metric("Período", f"{anos[0]}–{anos[1]}")
    c4.metric("TVs vendidas", f"{df[ALVO].sum():,}".replace(",", "."))  # 1,000 -> 1.000

    # Resumo por cidade: groupby agrupa e agg calcula média/soma de cada grupo
    st.subheader("Resumo por cidade")
    resumo = df.groupby("cidade").agg(
        preco_medio=("preco_medio", "mean"),
        media_vendas=(ALVO, "mean"),
        total_vendas=(ALVO, "sum"),
    ).sort_values("total_vendas", ascending=False)
    st.dataframe(resumo.round(2), width="stretch")

    # describe(): média, desvio padrão, mínimo, quartis e máximo de cada coluna numérica
    st.subheader("Estatísticas descritivas")
    # Sem coordenadas e sem o ano (média de ano/latitude não diz nada útil)
    numericas_analise = [c for c in df.select_dtypes("number").columns if c not in GEO + ["ano"]]
    st.dataframe(df[numericas_analise].describe().T.round(2), width="stretch")  # .T = transposta

# =====================================================================================
# 3. VISUALIZAÇÃO (Gustavo) — abas Gráficos e Mapa
# =====================================================================================
with aba_graficos:
    # Cada gráfico responde a uma pergunta; a legenda abaixo dele traz a conclusão

    # 1) Linha: tendência no tempo (plotly = interativo: zoom, hover, clique na legenda)
    por_ano = df.groupby(["ano", "cidade"], as_index=False)[ALVO].sum()
    st.plotly_chart(px.line(por_ano, x="ano", y=ALVO, color="cidade", markers=True,
                            title="As vendas estão crescendo?"), width="stretch")
    st.caption("Compare as linhas: se sobem ano a ano, o mercado está crescendo. "
               "O ano é uma informação importante para prever as vendas.")

    # 2) Barra (cidades) e pizza (marcas), lado a lado, com seaborn/matplotlib
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()  # cria a figura (fig) e a área do gráfico (ax)
        sns.barplot(data=por_cidade, x="cidade", y="vendas", ax=ax)
        ax.set_title("Onde se vende mais?")
        ax.tick_params(axis="x", rotation=30)  # gira os nomes das cidades
        st.pyplot(fig)                         # mostra a figura do matplotlib no app
        st.caption("Total vendido por cidade: mostra os mercados mais fortes.")
    with col2:
        # Pizza por marca (se a base não tiver marca, usa a cidade)
        grupo = "marca" if "marca" in df.columns else "cidade"
        fatias = df.groupby(grupo)[ALVO].sum()
        fig, ax = plt.subplots()
        ax.pie(fatias, labels=fatias.index, autopct="%1.1f%%")  # autopct: mostra a %
        ax.set_title(f"Qual {grupo} vende mais?")
        st.pyplot(fig)
        st.caption(f"Participação de cada {grupo} no total de TVs vendidas.")

    # 3) Preço: pd.cut divide os preços em 5 faixas; mostramos a média de vendas de cada uma
    faixas = pd.cut(df["preco_medio"], bins=5)
    por_faixa = df.groupby(faixas, observed=True)[ALVO].mean().reset_index()
    # Rótulo legível para cada faixa, ex.: "R$ 1.472 a 1.866"
    por_faixa["faixa de preço"] = por_faixa["preco_medio"].map(
        lambda f: f"R$ {f.left:,.0f} a {f.right:,.0f}".replace(",", "."))
    st.plotly_chart(px.bar(por_faixa, x="faixa de preço", y=ALVO,
                           title="O preço afeta as vendas?"), width="stretch")
    st.caption("Vendas médias por loja em cada faixa de preço: se as barras diminuem "
               "conforme o preço sobe, preço mais alto significa menos vendas.")
    # Dispersão: cada ponto é uma linha da base (loja em um ano)
    st.plotly_chart(px.scatter(df, x="preco_medio", y=ALVO, color="cidade",
                               title="Preço × vendas (cada ponto é uma loja em um ano)"),
                    width="stretch")

# ---------- Aba: Mapa (Gustavo) ----------
with aba_mapa:
    # Bolha por cidade, na latitude/longitude da base; tamanho e cor = total de vendas
    st.plotly_chart(px.scatter_map(por_cidade, lat="latitude", lon="longitude", size="vendas",
                                   color="vendas", hover_name="cidade", size_max=40, zoom=4,
                                   height=550, title="Total de TVs vendidas por cidade"),
                    width="stretch")

# =====================================================================================
# 4. MACHINE LEARNING (Guilherme) — aba Modelo e predição
# =====================================================================================
with aba_modelo:
    st.subheader("Modelo")
    # Entradas do modelo (features): todas as colunas menos o alvo e as coordenadas
    features = [c for c in base.columns if c not in [ALVO] + GEO]
    # Colunas de texto (cidade, marca) precisam virar números antes do treino
    categoricas = [c for c in features if not pd.api.types.is_numeric_dtype(base[c])]

    # Escolha do modelo pelo usuário
    escolha = st.selectbox("Algoritmo", ["Regressão Linear", "Random Forest"])
    if escolha == "Regressão Linear":
        regressor = LinearRegression()  # sem parâmetros para ajustar
    else:
        # Parâmetros ajustáveis: mais árvores = mais estável; mais profundidade = mais detalhe
        regressor = RandomForestRegressor(n_estimators=st.slider("Nº de árvores", 10, 300, 100),
                                          max_depth=st.slider("Profundidade máxima", 2, 20, 10),
                                          random_state=42)  # semente: resultado reproduzível

    # Treino: cada clique treina um modelo NOVO com a base atual (re-treino dinâmico)
    if st.button("Treinar / Re-treinar modelo"):
        X, y = base[features], base[ALVO]  # X = entradas, y = o que queremos prever
        # 80% para treinar, 20% guardados para testar em dados que o modelo nunca viu
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        # One-hot: "cidade" vira colunas 0/1 (cidade_Curitiba, cidade_Recife...).
        # handle_unknown="ignore": não quebra se aparecer uma categoria nova.
        # remainder="passthrough": colunas numéricas passam sem mudança.
        prep = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)],
            remainder="passthrough")
        # Pipeline: garante que a predição use exatamente o mesmo pré-processamento do treino
        modelo = Pipeline([("prep", prep), ("modelo", regressor)]).fit(X_tr, y_tr)
        pred = modelo.predict(X_te)  # previsões para os 20% de teste
        # Guarda no session_state para não perder no próximo clique
        st.session_state.modelo = modelo
        # R²: quanto da variação o modelo explica (1 = perfeito). MAE: erro médio em TVs
        st.session_state.metricas = (r2_score(y_te, pred), mean_absolute_error(y_te, pred))
        st.session_state.teste = pd.DataFrame({"real": y_te, "previsto": pred})  # para o gráfico

    # Tudo abaixo só aparece depois que existe um modelo treinado
    if "modelo" in st.session_state:
        r2, mae = st.session_state.metricas
        st.success(f"Modelo treinado — R²: {r2:.3f} | MAE: {mae:,.0f} TVs")

        col1, col2 = st.columns(2)
        # Gráfico real x previsto: pontos perto da diagonal = modelo acertando
        teste = st.session_state.teste
        fig = px.scatter(teste, x="real", y="previsto", title="O modelo acerta? (real × previsto)")
        fig.add_shape(type="line", x0=teste["real"].min(), y0=teste["real"].min(),
                      x1=teste["real"].max(), y1=teste["real"].max(), line_dash="dash")  # diagonal
        col1.plotly_chart(fig, width="stretch")
        col1.caption("Cada ponto é uma loja do conjunto de teste. Pontos perto da linha "
                     "tracejada = previsão próxima do valor real.")

        # Importância das variáveis: só o Random Forest calcula (feature_importances_)
        modelo_final = st.session_state.modelo["modelo"]  # pega o modelo de dentro do Pipeline
        if hasattr(modelo_final, "feature_importances_"):
            # Depois do one-hot os nomes viram "cat__cidade_Curitiba"; aqui voltam para "cidade"
            nomes = [n.split("__")[1] for n in st.session_state.modelo["prep"].get_feature_names_out()]
            nomes = [next((c for c in categoricas if n.startswith(c + "_")), n) for n in nomes]
            # Soma a importância de todas as colunas 0/1 da mesma variável
            importancia = (pd.Series(modelo_final.feature_importances_, index=nomes)
                           .groupby(level=0).sum().sort_values())
            col2.plotly_chart(px.bar(importancia, orientation="h", labels={"value": "importância",
                                     "index": "variável"}, title="O que mais pesa na previsão?")
                              .update_layout(showlegend=False), width="stretch")
            col2.caption("Confirma, pelo modelo, quais variáveis a análise dos gráficos "
                         "apontou como importantes.")
        else:
            col2.info("A importância das variáveis é exibida para o Random Forest.")

        # Predição: o formulário é montado automaticamente a partir das colunas da base
        st.subheader("Predição")
        entrada = {}
        for c in features:
            if c in categoricas:
                entrada[c] = st.selectbox(c, sorted(base[c].unique()))  # texto: lista de opções
            else:
                entrada[c] = st.number_input(c, value=float(base[c].median()))  # número: começa na mediana
        # O modelo recebe uma tabela de 1 linha e devolve a previsão
        previsto = st.session_state.modelo.predict(pd.DataFrame([entrada]))[0]
        st.metric("Vendas previstas (TVs/ano)", f"{previsto:,.0f}")
    else:
        st.info("Treine o modelo para habilitar predições.")

# ---------- Aba: Dados (Brian) ----------
with aba_dados:
    st.dataframe(df, width="stretch", hide_index=True)  # tabela filtrada, ordenável por coluna
