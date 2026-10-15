"""App Streamlit: análise de dados + ML com re-treino dinâmico."""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ARQUIVO = "dados.csv"  # base de dados persistida em disco
ALVO = "preco"         # coluna que o modelo aprende a prever

st.title("Previsão de preço de imóveis")

# ---------- Dados: carrega o CSV e permite adicionar novos dados ----------
# O session_state mantém o DataFrame entre as interações (o Streamlit
# reexecuta o script inteiro a cada clique).
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(ARQUIVO)

novo = st.sidebar.file_uploader("Adicionar novos dados (CSV)", type="csv")
if novo is not None and st.sidebar.button("Incorporar dados"):
    novos = pd.read_csv(novo)
    # Só aceita o CSV se tiver exatamente as mesmas colunas da base atual
    if set(novos.columns) == set(st.session_state.df.columns):
        # Junta os novos registros à base e grava no disco, assim os dados
        # (ex.: uma cidade nova) continuam lá na próxima execução
        st.session_state.df = pd.concat([st.session_state.df, novos], ignore_index=True)
        st.session_state.df.to_csv(ARQUIVO, index=False)
        st.sidebar.success(f"{len(novos)} linhas adicionadas. Re-treine o modelo.")
    else:
        st.sidebar.error(f"Colunas devem ser: {list(st.session_state.df.columns)}")

# Limpeza: remove linhas com valores faltantes e registros repetidos
df = st.session_state.df.dropna().drop_duplicates()

# ---------- Análise de dados ----------
st.header("Análise dos dados")
st.write(f"{len(df)} registros | cidades: {', '.join(sorted(df['cidade'].unique()))}")
st.dataframe(df.describe())  # estatísticas descritivas das colunas numéricas

# Dois gráficos lado a lado: preço por cidade e relação área x preço
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x="cidade", y=ALVO, ax=ax[0])
sns.scatterplot(data=df, x="area", y=ALVO, hue="cidade", ax=ax[1])
st.pyplot(fig)

# ---------- Configuração e (re)treino do modelo ----------
st.header("Modelo")
escolha = st.selectbox("Algoritmo", ["Regressão Linear", "Random Forest"])
params = {}  # hiperparâmetros, só usados no Random Forest
if escolha == "Random Forest":
    params["n_estimators"] = st.slider("Nº de árvores", 10, 300, 100)
    params["max_depth"] = st.slider("Profundidade máxima", 2, 20, 10)

# Cada clique treina um modelo novo do zero com todos os dados atuais,
# que é o que permite o re-treino depois de incorporar dados novos
if st.button("Treinar / Re-treinar modelo"):
    X, y = df.drop(columns=ALVO), df[ALVO]
    # 80% treino / 20% teste, para medir o desempenho em dados não vistos
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = LinearRegression() if escolha == "Regressão Linear" \
        else RandomForestRegressor(random_state=42, **params)
    # A cidade é texto, então vira colunas 0/1 (one-hot). handle_unknown evita
    # erro se aparecer uma cidade que não estava no treino. As demais colunas passam direto.
    prep = ColumnTransformer([("cidade", OneHotEncoder(handle_unknown="ignore"), ["cidade"])],
                             remainder="passthrough")
    # Pipeline junta pré-processamento + modelo, então a predição usa as mesmas transformações
    modelo = Pipeline([("prep", prep), ("modelo", regressor)]).fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    st.session_state.modelo = modelo
    # R² (quanto da variação o modelo explica) e MAE (erro médio em R$)
    st.session_state.metricas = (r2_score(y_te, pred), mean_absolute_error(y_te, pred))

if "metricas" in st.session_state:
    r2, mae = st.session_state.metricas
    st.success(f"Modelo treinado — R²: {r2:.3f} | MAE: R$ {mae:,.0f}")

    # ---------- Predição ----------
    st.header("Predição")
    cidade = st.selectbox("Cidade", sorted(df["cidade"].unique()))
    area = st.number_input("Área (m²)", 20, 500, 80)
    quartos = st.number_input("Quartos", 1, 10, 2)
    # Monta uma linha com as mesmas colunas usadas no treino (sem o alvo)
    entrada = pd.DataFrame([{"cidade": cidade, "area": area, "quartos": quartos}])
    st.metric("Preço previsto", f"R$ {st.session_state.modelo.predict(entrada)[0]:,.0f}")
else:
    st.info("Treine o modelo para habilitar predições.")
