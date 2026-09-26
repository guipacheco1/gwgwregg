"""Leitura flexível dos arquivos de entrada.

- Aceita .zip (com vários .txt/.csv dentro), .txt, .csv, .xlsx
- Detecta encoding (UTF-16 com BOM, UTF-8, Latin-1) e separador (; , TAB |)
- Identifica sozinho qual arquivo é o de PDV e qual é o de mercado (UF/cidade)
- Mapeia as colunas por sinônimos, para funcionar mesmo quando o layout muda um pouco
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Sinônimos de colunas (nome canônico -> nomes aceitos, já normalizados)
# ---------------------------------------------------------------------------
SINONIMOS: dict[str, list[str]] = {
    "pdv": ["CODIGO", "COD PDV", "CODIGO PDV", "ID PDV", "PDV", "COD CLIENTE", "CLIENTE", "CODIGO CLIENTE",
            "PDV CODE", "OUTLET", "OUTLET CODE", "OUTLET ID", "CUSTOMER", "CUSTOMER CODE", "CUSTOMER ID",
            "ACCOUNT", "ACCOUNT ID", "PONTO DE VENDA", "ESTABELECIMENTO"],
    "pdv_nome": ["DESCRICAO PDV", "NOME PDV", "RAZAO SOCIAL", "NOME CLIENTE",
                 "PDV NAME", "OUTLET NAME", "CUSTOMER NAME", "ACCOUNT NAME", "NOME FANTASIA",
                 "DESCRICAO"],
    "cnpj": ["CNPJ", "TAX ID"],
    "uf": ["ESTADO", "UF", "SIGLA UF", "STATE", "PROVINCE", "SIGLA ESTADO"],
    "cidade": ["CIDADE", "MUNICIPIO", "NOME CIDADE", "CITY", "TOWN"],
    "regiao": ["REGIAO", "REGION"],
    "canal": ["CANAL", "CANAL DETALHADO", "CHANNEL", "SUB CANAL", "SUBCANAL"],
    "mercado": ["MERCADO", "MERCADO RELEVANTE", "MARKET", "MKT"],
    "grupo": ["GRUPO", "GRUPO MERCADO", "CLASSE", "GROUP", "MARKET GROUP", "SUBMERCADO", "SUB MERCADO"],
    "periodo": ["PERIODO", "MES", "ANO MES", "DATA", "COMPETENCIA", "PERIOD", "MONTH", "DATE", "YEAR MONTH", "MES ANO"],
    "un": ["UN", "UNIDADES", "UNIDADE", "UNID", "QTD", "QTDE", "QUANTIDADE", "UNITS", "UNIT", "QTY", "QUANTITY",
           "VOLUME UN", "UN VENDIDAS"],
    "pf": ["PF", "VALOR", "VALOR PF", "RS", "VALOR RS", "PF RS", "VALUES", "VALUE", "SALES", "SALES VALUE",
           "VALOR VENDA", "FATURAMENTO", "PRECO FABRICA"],
    "fator": ["FATOR 2", "FATOR", "FACTOR", "PRIMO", "NUMERO PRIMO", "FATOR PRIMO"],
    "apresentacao": ["APRESENTACAO", "SKU", "DESCRICAO APRESENTACAO", "PRODUTO APRESENTACAO", "PRESENTATION",
                     "PACK", "PACK DESCRIPTION", "PRODUCT PACK", "SKU DESCRIPTION"],
    "produto": ["PRODUTO", "MARCA", "PRODUCT", "BRAND"],
}

# prefixos/sufixos típicos de nome "de sistema" (SG_UF, QTDE_UNID, VLR_PF, DT_PERIODO, COD_PDV_ID...)
_PREFIXOS = {"UN", "SG", "DS", "DESC", "DESCR", "DT", "DATA", "COD", "CD", "CODIGO", "VLR", "VL", "QTDE", "QTD", "QT",
             "NM", "NOME", "ID", "NR", "NUM", "TX", "TP", "TIPO", "FL", "SUM", "SOMA", "TOTAL", "TOT"}
_SUFIXOS = {"ID", "COD", "CODE", "DESC", "NOME", "NAME", "TOTAL"}


def _variantes(n: str) -> list[str]:
    """'COD PDV ID' -> ['PDV ID', 'COD PDV', 'PDV']."""
    t = n.split()
    out = []
    ini = 1 if len(t) > 1 and t[0] in _PREFIXOS else 0
    fim = len(t) - 1 if len(t) - ini > 1 and t[-1] in _SUFIXOS else len(t)
    for i, f in ((ini, len(t)), (0, fim), (ini, fim)):
        v = " ".join(t[i:f])
        if v and v != n and v not in out:
            out.append(v)
    return out


ROTULOS = {
    "pdv": "Código do PDV",
    "pdv_nome": "Nome do PDV",
    "cnpj": "CNPJ",
    "uf": "UF / Estado",
    "cidade": "Cidade",
    "regiao": "Região",
    "canal": "Canal",
    "mercado": "Mercado",
    "grupo": "Grupo",
    "periodo": "Período",
    "un": "Unidades",
    "pf": "Valor (PF)",
    "fator": "Fator / nº primo",
    "apresentacao": "Apresentação",
    "produto": "Produto",
}

# apresentação é opcional no arquivo de PDV: quando não existe, todos os PDVs são quebrados
OBRIG_PDV = ["pdv", "uf", "periodo", "un"]
OBRIG_MERC = ["uf", "periodo", "un", "apresentacao"]


def normaliza(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).strip().upper()
    return s


def mapear_colunas(colunas: list[str]) -> dict[str, str | None]:
    """Devolve {nome_canonico: coluna_original_ou_None}.

    Três passadas, da mais segura para a mais tolerante, para que um nome exato
    (ex.: CANAL) nunca perca para um aproximado (ex.: COD_CANAL):
      1) nome igual a um sinônimo
      2) nome sem prefixo/sufixo de sistema igual a um sinônimo (SG_UF -> UF)
      3) nome que começa com um sinônimo (FATOR_NOVO -> FATOR)
    """
    norm = {normaliza(c): c for c in colunas}
    variantes = {n: _variantes(n) for n in norm}
    usados: set[str] = set()
    resultado: dict[str, str | None] = {c: None for c in SINONIMOS}

    def tenta(teste):
        for canon, sins in SINONIMOS.items():
            if resultado[canon]:
                continue
            for s in sins:
                achou = next((orig for n, orig in norm.items() if orig not in usados and teste(n, s)), None)
                if achou:
                    resultado[canon] = achou
                    usados.add(achou)
                    break

    tenta(lambda n, s: n == s)
    tenta(lambda n, s: s in variantes[n])
    tenta(lambda n, s: len(s) >= 3 and n.startswith(s + " "))
    return resultado


# ---------------------------------------------------------------------------
# Leitura de arquivos
# ---------------------------------------------------------------------------
def _encoding(amostra: bytes) -> str:
    """Descobre o encoding só pelos primeiros bytes (o arquivo é lido em fluxo depois)."""
    if amostra[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    if amostra[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if amostra.count(b"\x00") > len(amostra) // 4:  # UTF-16 sem BOM
        return "utf-16-le"
    try:
        amostra[:-4].decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _separador(texto: str) -> str:
    primeira = texto.splitlines()[0] if texto else ""
    candidatos = [";", "\t", "|", ","]
    contagem = {c: primeira.count(c) for c in candidatos}
    melhor = max(contagem, key=contagem.get)
    if contagem[melhor] == 0:
        try:
            return csv.Sniffer().sniff(texto).delimiter
        except csv.Error:
            return ";"
    return melhor


def _espiar(abrir) -> tuple[str, str, list[str]]:
    """Lê só o começo do arquivo: encoding, separador e nomes das colunas."""
    with abrir() as fh:
        amostra = fh.read(64_000)
    enc = _encoding(amostra)
    texto = amostra.decode(enc, errors="ignore").lstrip("\ufeff")
    sep = _separador(texto)
    cab = next(csv.reader(io.StringIO(texto.splitlines()[0] if texto else ""), delimiter=sep), [])
    return enc, sep, [c.strip() for c in cab]


def _relevante(colunas: list[str]) -> bool:
    m = mapear_colunas(colunas)
    return bool(m["un"] or m["apresentacao"] or m["pf"] or m["fator"])


def ler_tabela(nome: str, abrir, so_relevantes: bool = False) -> pd.DataFrame | None:
    """abrir() devolve um arquivo binário. Lê em fluxo, sem carregar o texto inteiro na memória."""
    ext = nome.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xlsm", "xls"):
        with abrir() as fh:
            df = pd.read_excel(fh, dtype=str)
        return None if so_relevantes and not _relevante(list(df.columns)) else df
    enc, sep, cab = _espiar(abrir)
    if so_relevantes and not _relevante(cab):
        return None  # ex.: STAKEHOLDERS, que não entra na quebra
    for tentativa in (enc, "latin-1"):
        try:
            with abrir() as fh:
                df = pd.read_csv(fh, sep=sep, dtype=str, encoding=tentativa,
                                 keep_default_na=False, na_values=[""])
            break
        except UnicodeDecodeError:
            continue
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def abrir_arquivos(arquivos: list[tuple[str, bytes]]) -> dict[str, pd.DataFrame]:
    """Recebe [(nome, bytes)] - zip ou arquivos soltos - e devolve {nome: DataFrame}.
    Dentro de ZIP, arquivos sem colunas de venda (unidades/valor/apresentação) são ignorados."""
    tabelas: dict[str, pd.DataFrame] = {}
    for nome, raw in arquivos:
        if nome.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    if info.filename.lower().rsplit(".", 1)[-1] in ("txt", "csv", "xlsx", "xls", "tsv"):
                        df = ler_tabela(info.filename, lambda i=info: z.open(i), so_relevantes=True)
                        if df is not None:
                            tabelas[info.filename.split("/")[-1]] = df
        else:
            tabelas[nome] = ler_tabela(nome, lambda r=raw: io.BytesIO(r))
    return tabelas


@dataclass
class Identificacao:
    pdv_nome: str | None = None
    merc_nome: str | None = None
    pdv_map: dict = field(default_factory=dict)
    merc_map: dict = field(default_factory=dict)
    ignorados: list = field(default_factory=list)


def identificar(tabelas: dict[str, pd.DataFrame]) -> Identificacao:
    """Arquivo de PDV = tem código do PDV + unidades. Arquivo de mercado = unidades + apresentação, sem PDV."""
    ident = Identificacao()
    candidatos_pdv, candidatos_merc = [], []
    for nome, df in tabelas.items():
        m = mapear_colunas(list(df.columns))
        if m["un"] and m["pdv"]:
            candidatos_pdv.append((nome, m))
        elif m["un"] and m["apresentacao"]:
            candidatos_merc.append((nome, m))
        else:
            ident.ignorados.append(nome)
    # se houver mais de um candidato, fica com o maior
    candidatos_pdv.sort(key=lambda t: -len(tabelas[t[0]]))
    candidatos_merc.sort(key=lambda t: -len(tabelas[t[0]]))
    if candidatos_pdv:
        ident.pdv_nome, ident.pdv_map = candidatos_pdv[0][0], candidatos_pdv[0][1]
        ident.ignorados += [c[0] for c in candidatos_pdv[1:]]
    if candidatos_merc:
        ident.merc_nome, ident.merc_map = candidatos_merc[0][0], candidatos_merc[0][1]
        ident.ignorados += [c[0] for c in candidatos_merc[1:]]
    return ident


# ---------------------------------------------------------------------------
# Números no padrão BR ("1234,56") ou US ("1234.56", "1.2e+003")
# ---------------------------------------------------------------------------
_SCI = re.compile(r"^-?\d+(\.\d+)?[eE][+-]?\d+$")


def para_numero(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.strip().str.replace(r"[R$\s]", "", regex=True)
    tem_virgula = t.str.contains(",", regex=False).any()
    if tem_virgula:
        sci = t.str.match(_SCI)
        br = t.where(sci, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        t = br
    return pd.to_numeric(t, errors="coerce").fillna(0.0)
