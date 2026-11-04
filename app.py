"""App Streamlit: análise de vendas de TVs e prévia do modelo de ML."""
import pandas as pd
import plotly.express as px
import streamlit as st

ARQUIVO = "dados.csv"  # base de dados persistida em disco
ALVO = "vendas"        # coluna que o modelo vai aprender a prever

# O CSV não tem latitude/longitude, então as coordenadas do mapa vêm desta tabela
COORDENADAS = {
    "Curitiba": (-25.4284, -49.2733),
    "Florianopolis": (-27.5954, -48.5480),
    "Porto Alegre": (-30.0346, -51.2177),
    "Sao Paulo": (-23.5505, -46.6333),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Belo Horizonte": (-19.9167, -43.9345),
    "Brasilia": (-15.7939, -47.8828),
    "Salvador": (-12.9777, -38.5016),
    "Recife": (-8.0476, -34.8770),
    "Fortaleza": (-3.7319, -38.5267),
    "Manaus": (-3.1190, -60.0217),
    "Goiania": (-16.6869, -49.2648),
}


def ler_arquivo(arquivo):
    """Lê o arquivo enviado conforme a extensão (.csv, .json ou .xml)."""
    extensao = arquivo.name.rsplit(".", 1)[-1].lower()
    if extensao == "csv":
        return pd.read_csv(arquivo)
    if extensao == "json":
        return pd.read_json(arquivo)  # lista de registros: [{"cidade": ..., ...}, ...]
    # XML: <dados><registro>...</registro></dados>. O parser "etree" já vem no Python.
    return pd.read_xml(arquivo, parser="etree")


st.set_page_config(page_title="Vendas de TVs", layout="wide")
st.title("Previsão de vendas de TVs por ano")

# ---------- Dados ----------
# O session_state mantém o DataFrame entre as interações (o Streamlit
# reexecuta o script inteiro a cada clique).
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(ARQUIVO)

# Upload de novos dados na barra lateral
st.sidebar.header("Enviar novos dados")
novo = st.sidebar.file_uploader("Arquivo CSV, JSON ou XML", type=["csv", "json", "xml"])
if novo is not None and st.sidebar.button("Incorporar dados"):
    try:
        novos = ler_arquivo(novo)
        # Só aceita o arquivo se tiver as mesmas colunas da base atual
        if set(novos.columns) == set(st.session_state.df.columns):
            st.session_state.df = pd.concat([st.session_state.df, novos], ignore_index=True)
            st.session_state.df.to_csv(ARQUIVO, index=False)  # grava no disco
            st.sidebar.success(f"{len(novos)} linhas adicionadas.")
        else:
            st.sidebar.error(f"Colunas devem ser: {list(st.session_state.df.columns)}")
    except Exception as erro:  # arquivo corrompido ou fora do formato
        st.sidebar.error(f"Não foi possível ler o arquivo: {erro}")

# Limpeza: remove linhas com valores faltantes e registros repetidos
base = st.session_state.df.dropna().drop_duplicates()

# Filtros: tudo o que aparece nas abas respeita a seleção
st.sidebar.header("Filtros")
todas_cidades = sorted(base["cidade"].unique())
cidades = st.sidebar.multiselect("Cidades", todas_cidades, default=todas_cidades)
ano_min, ano_max = int(base["ano"].min()), int(base["ano"].max())
anos = st.sidebar.slider("Período", ano_min, ano_max, (ano_min, ano_max))

df = base[base["cidade"].isin(cidades) & base["ano"].between(*anos)]
if df.empty:
    st.warning("Nenhum dado para os filtros escolhidos.")
    st.stop()

# Totais por cidade, usados em várias abas
por_cidade = df.groupby("cidade", as_index=False)[ALVO].sum()

aba_geral, aba_graficos, aba_mapa, aba_modelo, aba_dados = st.tabs(
    ["Visão geral", "Gráficos", "Mapa", "Modelo (em breve)", "Dados"])

# ---------- Aba: Visão geral ----------
with aba_geral:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", len(df))
    c2.metric("Cidades", df["cidade"].nunique())
    c3.metric("Período", f"{anos[0]}–{anos[1]}")
    c4.metric("TVs vendidas", f"{df[ALVO].sum():,}".replace(",", "."))

    st.subheader("Resumo por cidade")
    resumo = df.groupby("cidade").agg(
        preco_medio=("preco_medio", "mean"),
        media_vendas=(ALVO, "mean"),
        total_vendas=(ALVO, "sum"),
    ).sort_values("total_vendas", ascending=False)
    st.dataframe(resumo.round(2), width="stretch")

    st.subheader("Estatísticas descritivas")
    st.dataframe(df[["preco_medio", ALVO]].describe().T.round(2), width="stretch")

# ---------- Aba: Gráficos (plotly = interativos) ----------
with aba_graficos:
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(por_cidade, x="cidade", y=ALVO, color="cidade",
                             title="Total de TVs vendidas por cidade"))
    col2.plotly_chart(px.pie(por_cidade, names="cidade", values=ALVO,
                             title="Participação por cidade"))

    por_ano = df.groupby(["ano", "cidade"], as_index=False)[ALVO].sum()
    st.plotly_chart(px.line(por_ano, x="ano", y=ALVO, color="cidade", markers=True,
                            title="TVs vendidas por ano"))

    # Preço x vendas: mostra que preço mais alto tende a vender menos
    st.plotly_chart(px.scatter(df, x="preco_medio", y=ALVO, color="cidade",
                               title="Preço médio × vendas por loja"))

# ---------- Aba: Mapa ----------
with aba_mapa:
    mapa = por_cidade[por_cidade["cidade"].isin(COORDENADAS)].copy()
    mapa["lat"] = mapa["cidade"].map(lambda c: COORDENADAS[c][0])
    mapa["lon"] = mapa["cidade"].map(lambda c: COORDENADAS[c][1])
    st.plotly_chart(px.scatter_map(mapa, lat="lat", lon="lon", size=ALVO, color=ALVO,
                                   hover_name="cidade", size_max=40, zoom=4, height=550,
                                   title="Total de TVs vendidas por cidade"))

# ---------- Aba: Modelo (em desenvolvimento) ----------
# Prévia da interface: controles desativados, nenhum modelo é treinado ainda.
with aba_modelo:
    st.subheader("Modelo")
    st.markdown(
        "- O usuário vai **escolher o algoritmo**: Regressão Linear ou Random Forest.\n"
        "- No Random Forest, vai poder ajustar o **nº de árvores** e a **profundidade máxima**.\n"
        "- O botão **Treinar / Re-treinar** vai treinar um modelo novo com todos os dados atuais "
        "(80% treino / 20% teste), inclusive os enviados pela barra lateral.\n"
        "- Depois do treino, o app vai mostrar as métricas **R²** (quanto da variação o modelo "
        "explica) e **MAE** (erro médio, em TVs)."
    )
    st.selectbox("Algoritmo", ["Regressão Linear", "Random Forest"], disabled=True)
    st.slider("Nº de árvores (Random Forest)", 10, 300, 100, disabled=True)
    st.slider("Profundidade máxima (Random Forest)", 2, 20, 10, disabled=True)
    st.button("Treinar / Re-treinar modelo", disabled=True)

    st.subheader("Predição")
    st.markdown("O usuário vai informar **cidade, ano e preço médio da TV** e o modelo "
                "vai estimar quantas TVs serão vendidas no ano.")
    st.selectbox("Cidade", todas_cidades, disabled=True)
    st.number_input("Ano", 2015, 2050, 2025, disabled=True)
    st.number_input("Preço médio da TV (R$)", 500, 10000, 2500, disabled=True)
    st.metric("Vendas previstas (TVs/ano)", "—")

# ---------- Aba: Dados ----------
with aba_dados:
    st.dataframe(df, width="stretch", hide_index=True)
