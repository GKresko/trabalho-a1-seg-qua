"""App Streamlit: upload de dados, análise visual e ML com treino dinâmico (vendas de TVs)."""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ALVO = "vendas"                       # coluna que o modelo aprende a prever
GEO = ["latitude", "longitude"]       # colunas usadas só no mapa, não no modelo
OBRIGATORIAS = ["cidade", ALVO] + GEO

st.set_page_config(page_title="Vendas de TVs", layout="wide")
st.title("Análise e previsão de vendas de TVs por ano")


def ler_arquivo(arquivo):
    """Lê CSV, JSON ou XML conforme a extensão do arquivo enviado."""
    nome = arquivo.name.lower()
    if nome.endswith(".csv"):
        return pd.read_csv(arquivo)
    if nome.endswith(".json"):
        return pd.read_json(arquivo)
    if nome.endswith(".xml"):
        return pd.read_xml(arquivo)
    raise ValueError("Formato não suportado (use .csv, .xml ou .json)")


# ---------- Upload: o usuário pode enviar outra base com as mesmas colunas ----------
arquivo = st.sidebar.file_uploader("Enviar base de dados", type=["csv", "json", "xml"])
if arquivo is not None:
    try:
        df = ler_arquivo(arquivo)
    except Exception as erro:
        st.sidebar.error(f"Erro ao ler o arquivo: {erro}")
        st.stop()
    faltando = [c for c in OBRIGATORIAS if c not in df.columns]
    if faltando:
        st.sidebar.error(f"Colunas obrigatórias ausentes: {faltando}")
        st.stop()
    st.sidebar.success(f"{arquivo.name} carregado")
else:
    df = pd.read_csv("dados.csv")  # base de exemplo
    st.sidebar.info("Usando a base de exemplo (dados.csv)")

# Se a base mudou (outro arquivo), descarta o modelo treinado antes
assinatura = (arquivo.name if arquivo else "exemplo", len(df))
if st.session_state.get("assinatura") != assinatura:
    st.session_state.assinatura = assinatura
    st.session_state.pop("modelo", None)
    st.session_state.pop("metricas", None)

# Limpeza: remove linhas com valores faltantes e registros repetidos
df = df.dropna().drop_duplicates()

# ---------- Análise de dados ----------
st.header("Análise dos dados")
st.write(f"{len(df)} registros | cidades: {', '.join(sorted(df['cidade'].unique()))}")
st.dataframe(df.describe())

# Total de vendas por cidade, usado em barra, pizza e mapa
por_cidade = df.groupby("cidade", as_index=False).agg(
    vendas=(ALVO, "sum"), latitude=("latitude", "mean"), longitude=("longitude", "mean"))

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.barplot(data=por_cidade, x="cidade", y="vendas", ax=ax)
    ax.set_title("Total de TVs vendidas por cidade")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    ax.pie(por_cidade["vendas"], labels=por_cidade["cidade"], autopct="%1.1f%%")
    ax.set_title("Participação por cidade")
    st.pyplot(fig)

# Mapa: cada ponto é uma cidade, o tamanho do círculo é o total de vendas
st.subheader("Mapa de vendas por cidade")
st.map(por_cidade, latitude="latitude", longitude="longitude", size="vendas")

# Gráficos interativos (plotly): zoom, hover e filtro pela legenda
por_ano = df.groupby(["ano", "cidade"], as_index=False)[ALVO].sum()
st.plotly_chart(px.line(por_ano, x="ano", y=ALVO, color="cidade", markers=True,
                        title="TVs vendidas por ano"), width="stretch")
st.plotly_chart(px.scatter(df, x="preco_medio", y=ALVO, color="cidade",
                           title="Preço médio x vendas"), width="stretch")

# ---------- Configuração e (re)treino do modelo ----------
st.header("Modelo")
# Variáveis do modelo: tudo exceto o alvo e as coordenadas
features = [c for c in df.columns if c not in [ALVO] + GEO]
categoricas = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
numericas = [c for c in features if c not in categoricas]

escolha = st.selectbox("Algoritmo", ["Regressão Linear", "Árvore de Decisão", "KNN", "Random Forest"])
# Cada algoritmo mostra só os seus parâmetros
if escolha == "Regressão Linear":
    regressor = LinearRegression()
elif escolha == "Árvore de Decisão":
    regressor = DecisionTreeRegressor(max_depth=st.slider("Profundidade máxima", 1, 20, 5), random_state=42)
elif escolha == "KNN":
    regressor = KNeighborsRegressor(n_neighbors=st.slider("Nº de vizinhos", 1, 30, 5))
else:
    regressor = RandomForestRegressor(n_estimators=st.slider("Nº de árvores", 10, 300, 100),
                                      max_depth=st.slider("Profundidade máxima", 2, 20, 10), random_state=42)

# Cada clique treina um modelo novo do zero com os dados atuais (re-treino dinâmico)
if st.button("Treinar / Re-treinar modelo"):
    X, y = df[features], df[ALVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    # Texto vira colunas 0/1 (one-hot); números são padronizados, o que importa para o KNN
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", StandardScaler(), numericas)])
    modelo = Pipeline([("prep", prep), ("modelo", regressor)]).fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    st.session_state.modelo = modelo
    st.session_state.metricas = (r2_score(y_te, pred), mean_absolute_error(y_te, pred))

if "modelo" in st.session_state:
    r2, mae = st.session_state.metricas
    st.success(f"Modelo treinado — R²: {r2:.3f} | MAE: {mae:,.0f} TVs")

    # ---------- Predição: o formulário é montado a partir das colunas da base ----------
    st.header("Predição")
    entrada = {}
    for c in features:
        if c in categoricas:
            entrada[c] = st.selectbox(c, sorted(df[c].unique()))
        else:
            entrada[c] = st.number_input(c, value=float(df[c].median()))
    previsto = st.session_state.modelo.predict(pd.DataFrame([entrada]))[0]
    st.metric("Vendas previstas (TVs/ano)", f"{previsto:,.0f}")
else:
    st.info("Treine o modelo para habilitar predições.")
