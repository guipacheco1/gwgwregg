"""Tracker - quebra dos PDVs sem apresentação identificada.

Rodar localmente:  streamlit run app.py
"""
import io
import time

import pandas as pd
import streamlit as st

from nucleo.leitura import OBRIG_MERC, OBRIG_PDV, ROTULOS, abrir_arquivos, identificar, normaliza
from nucleo.motor import Config, detectar_marcadores, executar, mascara_sem_apresentacao

st.set_page_config(page_title="Tracker", page_icon="📊", layout="wide")
st.title("📊 Tracker – Quebra por Apresentação")
st.caption("Suba os arquivos (ZIP ou o arquivo de PDV + o de mercado). O app calcula os restantes do mercado e "
           "quebra os PDVs sem apresentação identificada por meio de análise combinatória.")


# ---------------------------------------------------------------------------
# 1. Upload
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Lendo arquivos...", max_entries=1)
def carregar(arquivos: tuple):
    return abrir_arquivos(list(arquivos))


with st.sidebar:
    st.header("1. Arquivos")
    ups = st.file_uploader("ZIP ou arquivos .txt/.csv/.xlsx", accept_multiple_files=True,
                           type=["zip", "txt", "csv", "tsv", "xlsx", "xls"])
    caminho = st.text_input("...ou caminho do arquivo no computador (quando rodar localmente)",
                            placeholder=r"C:\pasta\arquivo.zip").strip().strip('"')

if not ups and not caminho:
    st.info("⬅️ Suba os arquivos na barra lateral para começar. Pode ser o ZIP inteiro "
            "— o app encontra sozinho o arquivo de PDV e o de mercado.")
    with st.expander("Como funciona"):
        st.markdown("""
1. **Restantes** = unidades do mercado (por geografia) − unidades já identificadas nos PDVs, por
   Período × Mercado × Grupo × Canal × Geografia × Apresentação.
2. **Loop** por cada combinação, no nível **UF**, **Cidade** ou **Região**.
3. Em cada combinação, a análise combinatória distribui os restantes entre os PDVs sem apresentação:
   - unidades de cada PDV batem exatamente;
   - o total por apresentação bate com o restante do mercado;
   - **valor (PF)** de cada PDV bate *(opcional)*;
   - **número primo (fator)** de cada PDV bate *(opcional)*.
4. No nível Cidade, cidades que não existem no mercado caem em *OUTRAS CIDADES &lt;UF&gt;*; PDVs que ficam sem
   mercado são reprocessados no nível UF com o saldo (cascata).
""")
    st.stop()

if ups:
    tabelas = carregar(tuple((u.name, u.getvalue()) for u in ups))
else:
    try:
        with open(caminho, "rb") as fh:
            tabelas = carregar(((caminho.replace("\\", "/").split("/")[-1], fh.read()),))
    except OSError as e:
        st.error(f"Não consegui abrir o arquivo: {e}")
        st.stop()
ident = identificar(tabelas)

# ---------------------------------------------------------------------------
# 2. Identificação e mapeamento de colunas
# ---------------------------------------------------------------------------
with st.sidebar:
    nomes = list(tabelas)
    pdv_nome = st.selectbox("Arquivo de PDV", nomes,
                           index=nomes.index(ident.pdv_nome) if ident.pdv_nome in nomes else 0)
    merc_nome = st.selectbox("Arquivo de mercado", nomes,
                           index=nomes.index(ident.merc_nome) if ident.merc_nome in nomes else min(1, len(nomes) - 1))

pdv_raw, merc_raw = tabelas[pdv_nome], tabelas[merc_nome]


@st.cache_data(show_spinner=False, max_entries=8)
def _mascara_desconhecidos(nome, col, marcador, nome_merc, col_merc, _df, _dfm):
    if not col:  # sem coluna de apresentação: todos os PDVs são quebrados
        return pd.Series(True, index=_df.index)
    return mascara_sem_apresentacao(_df[col], _dfm[col_merc], marcador)


@st.cache_data(show_spinner=False, max_entries=8)
def _marcadores(nome, col, nome_merc, col_merc, _df, _dfm):
    return detectar_marcadores(_df[col], _dfm[col_merc])


@st.cache_data(show_spinner=False, max_entries=8)
def _pct_outras(nome, col_cid, col_un, _df):
    from nucleo.leitura import para_numero
    cid = _df[col_cid].fillna("").astype(str).map(normaliza)
    un = para_numero(_df[col_un])
    return float(un[cid.str.startswith("OUTRAS")].sum() / max(un.sum(), 1))
from nucleo.leitura import mapear_colunas  # noqa: E402

auto_pdv, auto_merc = mapear_colunas(list(pdv_raw.columns)), mapear_colunas(list(merc_raw.columns))

with st.expander("🔧 Mapeamento de colunas (ajuste se o layout mudou)", expanded=False):
    st.caption("Detectado automaticamente pelos nomes. Se alguma coluna vier com outro nome, escolha aqui.")
    c1, c2 = st.columns(2)
    pdv_map, merc_map = {}, {}
    for canon, rot in ROTULOS.items():
        op_pdv = ["(nenhuma)"] + list(pdv_raw.columns)
        op_merc = ["(nenhuma)"] + list(merc_raw.columns)
        v = c1.selectbox(f"PDV · {rot}" + (" *" if canon in OBRIG_PDV else ""), op_pdv,
                         index=op_pdv.index(auto_pdv[canon]) if auto_pdv[canon] else 0, key=f"pdv_{canon}")
        pdv_map[canon] = None if v == "(nenhuma)" else v
        if canon in ("pdv", "pdv_nome", "cnpj"):
            merc_map[canon] = None
            continue
        v = c2.selectbox(f"Mercado · {rot}" + (" *" if canon in OBRIG_MERC else ""), op_merc,
                         index=op_merc.index(auto_merc[canon]) if auto_merc[canon] else 0, key=f"merc_{canon}")
        merc_map[canon] = None if v == "(nenhuma)" else v

faltando = [ROTULOS[c] for c in OBRIG_PDV if not pdv_map.get(c)] + [ROTULOS[c] for c in OBRIG_MERC if not merc_map.get(c)]
if faltando:
    st.error("Colunas obrigatórias não encontradas: " + ", ".join(faltando) + ". Ajuste no mapeamento acima.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Linhas PDV", f"{len(pdv_raw):,}".replace(",", "."))
m2.metric("Linhas mercado", f"{len(merc_raw):,}".replace(",", "."))

# ---------------------------------------------------------------------------
# 3. Parâmetros
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("2. Parâmetros")
    if pdv_map.get("apresentacao"):
        marcador = st.text_input("Valor que indica PDV sem apresentação", "",
                                 placeholder="Automático",
                                 help="Deixe vazio para detectar sozinho: linhas vazias ou com apresentação "
                                      "que não existe no arquivo de mercado.")
        if not marcador.strip():
            achados = _marcadores(pdv_nome, pdv_map["apresentacao"], merc_nome, merc_map["apresentacao"],
                                  _df=pdv_raw, _dfm=merc_raw)
            if achados:
                st.caption("Detectado automaticamente: " + ", ".join(f"`{a}`" for a in achados))
    else:
        marcador = ""
        st.info("O arquivo de PDV não tem coluna de apresentação: todos os PDVs serão quebrados "
                "usando as apresentações do arquivo de mercado.")

    desc = _mascara_desconhecidos(pdv_nome, pdv_map.get("apresentacao"), marcador,
                                  merc_nome, merc_map["apresentacao"], _df=pdv_raw, _dfm=merc_raw)
    m3.metric("Linhas sem apresentação", f"{int(desc.sum()):,}".replace(",", "."))

    nivel_rot = st.radio("Nível geográfico do loop (contratado pelo laboratório)",
                         ["UF", "Cidade", "Região"], horizontal=True,
                         help="UF: loop por estado. Cidade: cidades fora do arquivo de mercado viram 'OUTRAS CIDADES <UF>'.")
    nivel = {"UF": "uf", "Cidade": "cidade", "Região": "regiao"}[nivel_rot]
    rotulo_outras = "OUTRAS CIDADES {uf}"
    cascata = True
    if nivel == "cidade":
        rotulo_outras = st.text_input("Rótulo do 'resto do estado' no arquivo de mercado", "OUTRAS CIDADES {uf}")
        cascata = st.checkbox("Cascata para UF (PDVs sem mercado na cidade usam o saldo da UF)", True)

    st.subheader("Análise combinatória")
    st.checkbox("Unidades", True, disabled=True, help="Sempre usado: a soma por PDV bate exatamente.")
    usar_valor = st.checkbox("Valor (PF)", True, disabled=not pdv_map.get("pf"))
    usar_primo = st.checkbox("Número primo (fator)", False, disabled=not pdv_map.get("fator"),
                             help="Usa o fator primo por apresentação (ex.: 0,07 / 0,11 / 0,19...) para identificar o SKU.")

    st.header("3. Filtros")
    st.caption("Dica: teste primeiro uma única combinação.")

    def opcoes(col_canon):
        col = pdv_map.get(col_canon)
        if not col:
            return []
        return sorted(pdv_raw.loc[desc, col].dropna().astype(str).str.strip().unique())

    f_per = st.multiselect("Período", opcoes("periodo"), placeholder="Todos")
    f_mer = st.multiselect("Mercado", [m.upper() for m in opcoes("mercado")], placeholder="Todos")
    f_gru = st.multiselect("Grupo", [g.upper() for g in opcoes("grupo")], placeholder="Todos")
    f_can = st.multiselect("Canal", opcoes("canal"), placeholder="Todos")
    f_uf = st.multiselect("UF", [u.upper() for u in opcoes("uf")], placeholder="Todas")

    with st.expander("Avançado"):
        inteiro = st.checkbox("Unidades inteiras", True)
        tempo = st.number_input("Tempo máx. por combinação (s)", 1.0, 120.0, 5.0)
        p_merc = st.number_input("Peso: bater total do mercado", 0.0, 1000.0, 10.0)
        p_val = st.number_input("Peso: bater valor PF", 0.0, 1000.0, 1.0)
        p_pri = st.number_input("Peso: bater número primo", 0.0, 1000.0, 10.0)

    rodar = st.button("▶️ Quebrar", type="primary", use_container_width=True)

# Diagnóstico do nível geográfico
if merc_map.get("cidade"):
    pct_outras = _pct_outras(merc_nome, merc_map["cidade"], merc_map["un"], _df=merc_raw)
    m4.metric("% do mercado em 'OUTRAS CIDADES'", f"{pct_outras:.0%}",
              help="Quanto maior, mais o mercado está no nível UF. Perto de 100% → use o nível UF.")

# ---------------------------------------------------------------------------
# 4. Execução
# ---------------------------------------------------------------------------
if rodar:
    cfg = Config(nivel_geo=nivel, usar_valor=usar_valor, usar_primo=usar_primo, marcador_concorrente=marcador,
                 rotulo_outras=rotulo_outras, inteiro=inteiro, cascata_uf=cascata, tempo_max_grupo=tempo,
                 periodos=f_per or None, mercados=f_mer or None, grupos=f_gru or None,
                 canais=f_can or None, ufs=f_uf or None,
                 peso_mercado=p_merc, peso_valor=p_val, peso_primo=p_pri)
    barra = st.progress(0.0, "Preparando...")
    t0 = time.time()
    res = executar(pdv_raw, pdv_map, merc_raw, merc_map, cfg, progresso=lambda p, m: barra.progress(min(p, 1.0), m))
    barra.progress(1.0, f"Concluído em {time.time() - t0:.1f} s")
    st.session_state["res"] = res
    st.session_state["cfg"] = cfg
    st.session_state["nivel_pedido"] = nivel

res = st.session_state.get("res")
if res is None:
    st.stop()

# ---------------------------------------------------------------------------
# 5. Resultados
# ---------------------------------------------------------------------------
r = res.resumo
st.subheader("Resultado")
nivel_pedido = st.session_state.get("nivel_pedido")
if nivel_pedido and nivel_pedido != r["nivel"]:
    st.info(f"O arquivo não tem a coluna necessária para o nível **{nivel_pedido}** nos dois arquivos (PDV e mercado); "
            f"a quebra foi feita no nível **{r['nivel'].upper()}**.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Combinações", r["combinacoes"])
k2.metric("Exatas", f'{r["exatas"]} ({r["exatas"] / max(r["combinacoes"], 1):.0%})')
k3.metric("PDVs sem apresentação", r["pdv_desconhecidos"])
k4.metric("Unidades alocadas", f'{r["un_alocadas"]:,.0f} / {r["un_desconhecidas"]:,.0f}'.replace(",", "."))

aba1, aba2, aba3, aba4, aba5 = st.tabs(["Quebra por PDV", "Matriz PDV × apresentação", "Log das combinações",
                                        "Restantes do mercado", "Fator / preço médio"])
with aba1:
    st.dataframe(res.alocacao, use_container_width=True, height=450)
with aba2:
    st.dataframe(res.matriz, use_container_width=True, height=450)
with aba3:
    st.caption("'Aproximado' = não foi possível bater todas as restrições exatamente.")
    so_aprox = st.checkbox("Mostrar só as aproximadas / não alocadas")
    lg = res.log
    if so_aprox and len(lg):
        lg = lg[lg["qualidade"] != "Exato"]
    st.dataframe(lg, use_container_width=True, height=450)
with aba4:
    st.dataframe(res.restantes, use_container_width=True, height=450)
with aba5:
    st.dataframe(res.precos, use_container_width=True, height=450)


@st.cache_data(show_spinner="Gerando Excel...")
def _excel(_res, chave):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        _res.alocacao.head(1_000_000).to_excel(xw, sheet_name="Quebra_PDV", index=False)
        _res.matriz.head(1_000_000).to_excel(xw, sheet_name="Matriz", index=False)
        _res.log.to_excel(xw, sheet_name="Log", index=False)
        _res.restantes.head(1_000_000).to_excel(xw, sheet_name="Restantes", index=False)
        _res.precos.to_excel(xw, sheet_name="Fator_Preco", index=False)
    return buf.getvalue()


st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
chave = id(res)
d1.download_button("📊 Resultado completo (.xlsx)", _excel(res, chave), "tracker_quebra.xlsx",
                   use_container_width=True)
d2.download_button("📄 Matriz PDV × apresentação (.csv)", res.matriz.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                   "matriz_pdv_apresentacao.csv", use_container_width=True)
d3.download_button("🏥 Arquivo de PDV quebrado (.csv)",
                   res.pdv_final.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                   "pdv_quebrado.csv", use_container_width=True,
                   help="Arquivo de PDV original com as linhas sem apresentação substituídas pelas apresentações estimadas.")
