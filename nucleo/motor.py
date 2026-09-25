"""Motor de quebra por apresentação (Tracker).

1. Mercado (por geografia) - PDVs com apresentação já identificada = unidades RESTANTES
   por Período x Mercado x Grupo x Canal x Geografia x Apresentação.
2. Fator (nº primo) por apresentação = soma(fator) / soma(unidades)
   Preço médio por apresentação      = soma(valor) / soma(unidades)
3. Para cada combinação, distribui os restantes entre os PDVs sem apresentação
   resolvendo um problema inteiro (MILP / HiGHS):
      - unidades de cada PDV batem exatamente                (restrição dura)
      - total por apresentação bate com o restante do mercado (meta)
      - valor PF de cada PDV bate                              (meta, opcional)
      - fator (nº primo) de cada PDV bate                      (meta, opcional)
   As metas viram desvios penalizados, então o modelo SEMPRE tem solução; quando
   existe a solução exata, ela é encontrada com desvio zero.
4. Geografia do loop: UF, Cidade (cidades que não existem no mercado caem em
   "OUTRAS CIDADES <UF>") ou Região. PDVs que ficam sem mercado no nível escolhido
   são reprocessados no nível UF com o saldo que sobrou (cascata).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .leitura import normaliza, para_numero


@dataclass
class Config:
    nivel_geo: str = "uf"                # "uf" | "cidade" | "regiao"
    usar_valor: bool = True              # validar PF (valor)
    usar_primo: bool = False             # validar fator (número primo)
    marcador_concorrente: str = ""       # vazio = detectar automaticamente
    rotulo_outras: str = "OUTRAS CIDADES {uf}"
    inteiro: bool = True
    cascata_uf: bool = True
    tempo_max_grupo: float = 5.0        # segundos por combinação
    periodos: list | None = None         # None = todos
    mercados: list | None = None
    grupos: list | None = None
    canais: list | None = None
    ufs: list | None = None
    peso_mercado: float = 10.0
    peso_valor: float = 1.0
    peso_primo: float = 10.0
    tolerancia_primo: float = 0.011
    tolerancia_valor_pct: float = 0.01   # 1% do PF do PDV


@dataclass
class Resultado:
    alocacao: pd.DataFrame               # formato longo: PDV x apresentação
    matriz: pd.DataFrame                 # PDV x apresentações em colunas
    log: pd.DataFrame                    # uma linha por combinação do loop
    restantes: pd.DataFrame              # restantes do mercado por combinação
    precos: pd.DataFrame                 # fator e preço médio por apresentação
    pdv_final: pd.DataFrame               # arquivo de PDV com as linhas sem apresentação substituídas
    resumo: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preparação
# ---------------------------------------------------------------------------
def _padroniza(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for canon, col in mapa.items():
        if col is not None and col in df.columns:
            out[canon] = df[col]
    for c in ("un", "pf", "fator"):
        if c in out:
            out[c] = para_numero(out[c])
    for c in ("uf", "cidade", "regiao", "canal", "mercado", "grupo", "periodo", "apresentacao", "pdv"):
        if c in out:
            out[c] = out[c].fillna("").astype(str).str.strip()
    for c in ("uf", "cidade", "mercado", "grupo"):
        if c in out:
            out[c] = out[c].str.upper()
    return out


def _chaves_base(tp: pd.DataFrame, tm: pd.DataFrame) -> list[str]:
    """Colunas que definem o loop, presentes nos dois arquivos."""
    return [c for c in ("periodo", "mercado", "grupo", "canal") if c in tp and c in tm]


def _geo_cidade(tp: pd.DataFrame, tm: pd.DataFrame, rotulo: str) -> tuple[pd.Series, pd.Series]:
    cidades_merc = set(zip(tm["uf"], tm["cidade"].map(normaliza)))
    merc_geo = tm["cidade"].map(normaliza)
    pdv_norm = tp["cidade"].map(normaliza)
    tem = pd.Series([(u, c) in cidades_merc for u, c in zip(tp["uf"], pdv_norm)], index=tp.index)
    outras = tp["uf"].map(lambda u: normaliza(rotulo.format(uf=u)))
    pdv_geo = pdv_norm.where(tem, outras)
    return pdv_geo, merc_geo


def mascara_sem_apresentacao(ap_pdv: pd.Series, ap_merc: pd.Series, marcador: str = "") -> pd.Series:
    """True para as linhas do arquivo de PDV cuja apresentação precisa ser estimada.

    Com marcador informado: linhas vazias ou iguais ao marcador.
    Sem marcador (automático): linhas vazias ou cuja apresentação não existe no
    arquivo de mercado (ex.: um rótulo genérico usado para os PDVs sem apresentação).
    """
    ap = ap_pdv.fillna("").astype(str).map(normaliza)
    vazio = ap.isin(["", "0"])
    if marcador and marcador.strip():
        return vazio | (ap == normaliza(marcador))
    conhecidas = set(ap_merc.fillna("").astype(str).map(normaliza))
    return vazio | ~ap.isin(conhecidas)


def detectar_marcadores(ap_pdv: pd.Series, ap_merc: pd.Series, n: int = 3) -> list[str]:
    """Valores do arquivo de PDV que não existem como apresentação no mercado (os mais frequentes)."""
    m = mascara_sem_apresentacao(ap_pdv, ap_merc)
    v = ap_pdv[m].fillna("").astype(str).str.strip()
    return [x if x else "(vazio)" for x in v.value_counts().head(n).index]


def preparar(pdv_raw, pdv_map, merc_raw, merc_map, cfg: Config):
    tp = _padroniza(pdv_raw, pdv_map)
    tm = _padroniza(merc_raw, merc_map)
    if "fator" not in tp or "fator" not in tm:
        cfg.usar_primo = False
    if "pf" not in tp or "pf" not in tm:
        cfg.usar_valor = False
    for d in (tp, tm):
        if "pf" not in d:
            d["pf"] = 0.0
        if "fator" not in d:
            d["fator"] = d["un"]

    ap_merc_todas = tm["apresentacao"].copy()  # antes dos filtros, para a detecção automática

    # filtros
    for campo, sel in (("periodo", cfg.periodos), ("mercado", cfg.mercados), ("grupo", cfg.grupos),
                       ("canal", cfg.canais), ("uf", cfg.ufs)):
        if sel:
            if campo in tp:
                tp = tp[tp[campo].isin(sel)]
            if campo in tm:
                tm = tm[tm[campo].isin(sel)]

    # geografia
    tp["geo_uf"] = tp["uf"]
    tm["geo_uf"] = tm["uf"]
    nivel = cfg.nivel_geo
    if nivel == "cidade" and "cidade" in tp and "cidade" in tm:
        tp["geo"], tm["geo"] = _geo_cidade(tp, tm, cfg.rotulo_outras)
        tp["geo"] = tp["uf"] + " | " + tp["geo"]
        tm["geo"] = tm["uf"] + " | " + tm["geo"]
    elif nivel == "regiao" and "regiao" in tp and "regiao" in tm:
        tp["geo"] = tp["uf"] + " | " + tp["regiao"].map(normaliza)
        tm["geo"] = tm["uf"] + " | " + tm["regiao"].map(normaliza)
    else:
        nivel = "uf"
        tp["geo"], tm["geo"] = tp["uf"], tm["uf"]
    cfg.nivel_geo = nivel

    if "apresentacao" not in tp:  # sem coluna de apresentação: todo PDV precisa ser quebrado
        tp["apresentacao"] = ""
    tp["desconhecido"] = mascara_sem_apresentacao(tp["apresentacao"], ap_merc_todas, cfg.marcador_concorrente)
    return tp, tm


def tabela_precos(tm: pd.DataFrame, base: list[str]) -> pd.DataFrame:
    """Fator (nº primo) e preço médio por apresentação."""
    chave = [c for c in ("periodo", "mercado", "grupo") if c in base] + ["apresentacao"]
    g = tm.groupby(chave, dropna=False)[["un", "pf", "fator"]].sum().reset_index()
    g = g[g["un"] > 0]
    g["fator_un"] = (g["fator"] / g["un"]).round(6)
    g["preco_medio"] = g["pf"] / g["un"]
    return g[chave + ["fator_un", "preco_medio"]]


def calcular_restantes(tp, tm, base, geo_col="geo") -> pd.DataFrame:
    """Restantes: mercado - PDVs já identificados."""
    chave = base + [geo_col, "apresentacao"]
    merc = tm.groupby(chave, dropna=False)[["un", "pf", "fator"]].sum()
    ident = tp[~tp["desconhecido"]].groupby(chave, dropna=False)[["un", "pf", "fator"]].sum()
    r = merc.sub(ident, fill_value=0).reset_index()
    r = r.rename(columns={"un": "un_rest", "pf": "pf_rest", "fator": "fator_rest"})
    return r


# ---------------------------------------------------------------------------
# Otimização de uma combinação
# ---------------------------------------------------------------------------
def resolver_combo(pdvs: pd.DataFrame, prods: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, dict]:
    """pdvs: colunas un, pf, fator | prods: un_rest, fator_un, preco_medio.
    Retorna matriz X (n_pdv x n_prod) de unidades e métricas."""
    n, m = len(pdvs), len(prods)
    U = pdvs["un"].to_numpy(float)
    PF = pdvs["pf"].to_numpy(float)
    F2 = pdvs["fator"].to_numpy(float)
    R = np.maximum(prods["un_rest"].to_numpy(float), 0)
    f = prods["fator_un"].to_numpy(float)
    p = prods["preco_medio"].to_numpy(float)

    if m == 1:  # só uma apresentação possível: atribuição direta
        X = U.reshape(-1, 1).copy()
        return X, _metricas(X, pdvs, prods, cfg, "Direto (1 apresentação)")

    nx = n * m
    # variáveis: X (nx) | desvio mercado +/- (2m) | desvio valor +/- (2n) | desvio primo +/- (2n)
    usa_v, usa_f = cfg.usar_valor, cfg.usar_primo
    nv = nx + 2 * m + (2 * n if usa_v else 0) + (2 * n if usa_f else 0)
    c = np.zeros(nv)
    off = nx
    c[off:off + 2 * m] = cfg.peso_mercado
    idx_merc = off
    off += 2 * m
    esc_preco = max(np.mean(p[p > 0]) if (p > 0).any() else 1.0, 1e-9)
    if usa_v:
        idx_val = off
        c[off:off + 2 * n] = cfg.peso_valor / esc_preco
        off += 2 * n
    if usa_f:
        idx_pri = off
        fpos = f[f > 0]
        esc_f = max(fpos.min() if len(fpos) else 1.0, 1e-9)
        c[off:off + 2 * n] = cfg.peso_primo / esc_f
        off += 2 * n
    # desempate leve: prefere apresentações com maior restante (evita soluções "estranhas")
    c[:nx] = 1e-6

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    # 1) soma por PDV = unidades do PDV (dura)
    for i in range(n):
        for j in range(m):
            rows.append(r); cols.append(i * m + j); vals.append(1.0)
        lo.append(U[i]); hi.append(U[i]); r += 1
    # 2) soma por apresentação + d- - d+ = restante
    for j in range(m):
        for i in range(n):
            rows.append(r); cols.append(i * m + j); vals.append(1.0)
        rows += [r, r]; cols += [idx_merc + 2 * j, idx_merc + 2 * j + 1]; vals += [1.0, -1.0]
        lo.append(R[j]); hi.append(R[j]); r += 1
    # 3) valor PF por PDV
    if usa_v:
        for i in range(n):
            for j in range(m):
                rows.append(r); cols.append(i * m + j); vals.append(p[j])
            rows += [r, r]; cols += [idx_val + 2 * i, idx_val + 2 * i + 1]; vals += [1.0, -1.0]
            lo.append(PF[i]); hi.append(PF[i]); r += 1
    # 4) número primo (fator) por PDV
    if usa_f:
        for i in range(n):
            for j in range(m):
                rows.append(r); cols.append(i * m + j); vals.append(f[j])
            rows += [r, r]; cols += [idx_pri + 2 * i, idx_pri + 2 * i + 1]; vals += [1.0, -1.0]
            lo.append(F2[i]); hi.append(F2[i]); r += 1

    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    ub = np.full(nv, np.inf)
    ub[:nx] = np.repeat(U, m)
    integ = np.zeros(nv)
    if cfg.inteiro:
        integ[:nx] = 1
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), ub),
               options={"time_limit": cfg.tempo_max_grupo, "mip_rel_gap": 1e-6, "disp": False})
    if res.x is None:  # não deveria acontecer (modelo sempre viável); fallback proporcional
        share = R / R.sum() if R.sum() > 0 else np.full(m, 1 / m)
        X = np.outer(U, share)
        status = "Fallback proporcional"
    else:
        X = res.x[:nx].reshape(n, m)
        if cfg.inteiro:
            X = np.round(X)
        status = "Ótimo" if res.status == 0 else "Limite de tempo"
    return X, _metricas(X, pdvs, prods, cfg, status)


def _metricas(X, pdvs, prods, cfg, status):
    R = np.maximum(prods["un_rest"].to_numpy(float), 0)
    f = prods["fator_un"].to_numpy(float)
    p = prods["preco_medio"].to_numpy(float)
    dev_merc = float(np.abs(X.sum(0) - R).sum())
    dev_val = np.abs(X @ p - pdvs["pf"].to_numpy(float))
    dev_pri = np.abs(X @ f - pdvs["fator"].to_numpy(float))
    ok_val = (dev_val <= np.maximum(cfg.tolerancia_valor_pct * pdvs["pf"].to_numpy(float), 1.0)).mean()
    ok_pri = (dev_pri <= cfg.tolerancia_primo).mean()
    exato = dev_merc < 0.5 and (not cfg.usar_valor or ok_val == 1) and (not cfg.usar_primo or ok_pri == 1)
    return {
        "status": status,
        "qualidade": "Exato" if exato else "Aproximado",
        "desvio_un_mercado": dev_merc,
        "desvio_pf_total": float(dev_val.sum()),
        "pct_pdv_valor_ok": float(ok_val),
        "desvio_primo_total": float(dev_pri.sum()),
        "pct_pdv_primo_ok": float(ok_pri),
    }


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
def executar(pdv_raw, pdv_map, merc_raw, merc_map, cfg: Config,
             progresso: Callable[[float, str], None] | None = None) -> Resultado:
    tp, tm = preparar(pdv_raw, pdv_map, merc_raw, merc_map, cfg)
    base = _chaves_base(tp, tm)
    precos = tabela_precos(tm, base)
    chave_preco = [c for c in ("periodo", "mercado", "grupo") if c in base] + ["apresentacao"]

    niveis = [("geo", cfg.nivel_geo)]
    if cfg.cascata_uf and cfg.nivel_geo != "uf":
        niveis.append(("geo_uf", "uf"))

    desconhecidos = tp[tp["desconhecido"] & (tp["un"] > 0)].copy()
    desconhecidos["_id"] = np.arange(len(desconhecidos))
    alocs, logs = [], []
    pendentes = desconhecidos
    restantes_saida = None
    ja_alocado = None  # alocação da etapa anterior (para calcular saldo na cascata)

    for etapa, (geo_col, nome_nivel) in enumerate(niveis):
        rest = calcular_restantes(tp, tm, base, geo_col)
        if ja_alocado is not None and len(ja_alocado):
            usado = ja_alocado.groupby(base + [geo_col, "apresentacao"])["un_estimada"].sum()
            rest = rest.set_index(base + [geo_col, "apresentacao"])
            rest["un_rest"] = rest["un_rest"].sub(usado, fill_value=0)
            rest = rest.reset_index()
        rest = rest.merge(precos, on=chave_preco, how="left")
        rest["fator_un"] = rest["fator_un"].fillna(0)
        rest["preco_medio"] = rest["preco_medio"].fillna(
            (rest["pf_rest"] / rest["un_rest"].where(rest["un_rest"] > 0)).fillna(0))
        rest_pos = rest[rest["un_rest"] > 0.5]
        if restantes_saida is None:
            restantes_saida = rest_pos.copy()
            restantes_saida.insert(0, "nivel", nome_nivel)

        chave = base + [geo_col]
        grupos_rest = {k: g for k, g in rest_pos.groupby(chave, dropna=False)}
        combos = list(pendentes.groupby(chave, dropna=False))
        sem_mercado_ids = []
        total = len(combos)
        for k, (key, pdvs) in enumerate(combos):
            if progresso and (k % 25 == 0 or k == total - 1):
                progresso((k + 1) / max(total, 1), f"Etapa {etapa + 1} ({nome_nivel}): combinação {k + 1} de {total}")
            key = key if isinstance(key, tuple) else (key,)
            prods = grupos_rest.get(key)
            rotulo = dict(zip(chave, key))
            if prods is None or prods.empty:
                sem_mercado_ids += list(pdvs["_id"])
                if etapa == len(niveis) - 1:
                    logs.append({**rotulo, "nivel": nome_nivel, "n_pdv": len(pdvs), "n_apresentacoes": 0,
                                 "un_pdvs": pdvs["un"].sum(), "un_restantes": 0,
                                 "status": "Sem mercado restante", "qualidade": "Não alocado"})
                continue
            prods = prods.reset_index(drop=True)
            X, met = resolver_combo(pdvs.reset_index(drop=True), prods, cfg)
            logs.append({**rotulo, "nivel": nome_nivel, "n_pdv": len(pdvs), "n_apresentacoes": len(prods),
                         "un_pdvs": pdvs["un"].sum(), "un_restantes": prods["un_rest"].sum(), **met})
            ii, jj = np.nonzero(X > 1e-9)
            if len(ii):
                p_rows = pdvs.iloc[ii]
                a = pd.DataFrame({
                    "_id": p_rows["_id"].to_numpy(),
                    "apresentacao_estimada": prods["apresentacao"].to_numpy()[jj],
                    "un_estimada": X[ii, jj],
                    "fator_un": prods["fator_un"].to_numpy()[jj],
                    "preco_medio": prods["preco_medio"].to_numpy()[jj],
                    "nivel": nome_nivel,
                })
                alocs.append(a)
        # próxima etapa: só quem ficou sem mercado
        if alocs:
            tmp = pd.concat(alocs).merge(desconhecidos[["_id"] + base + ["geo", "geo_uf"]], on="_id")
            ja_alocado = tmp.rename(columns={"apresentacao_estimada": "apresentacao"})
        pendentes = desconhecidos[desconhecidos["_id"].isin(sem_mercado_ids)]
        if pendentes.empty:
            break

    aloc = pd.concat(alocs, ignore_index=True) if alocs else pd.DataFrame(
        columns=["_id", "apresentacao_estimada", "un_estimada", "fator_un", "preco_medio", "nivel"])
    aloc["pf_estimado"] = aloc["un_estimada"] * aloc["preco_medio"]
    aloc["fator_estimado"] = aloc["un_estimada"] * aloc["fator_un"]

    # anexa atributos originais do PDV
    orig = pdv_raw.loc[desconhecidos.index].copy()
    orig["_id"] = desconhecidos["_id"].to_numpy()
    orig["_geo_quebra"] = desconhecidos["geo"].to_numpy()
    alocacao = orig.merge(aloc, on="_id", how="right").drop(columns=["_id"])

    # matriz PDV x apresentação
    pdv_col = pdv_map.get("pdv")
    idx = [c for c in [pdv_col, pdv_map.get("periodo"), pdv_map.get("mercado"), pdv_map.get("grupo"),
                       pdv_map.get("canal"), pdv_map.get("uf"), "_geo_quebra"] if c]
    matriz = (alocacao.pivot_table(index=idx, columns="apresentacao_estimada", values="un_estimada",
                                   aggfunc="sum", fill_value=0).reset_index()) if len(alocacao) else pd.DataFrame()

    # arquivo final: linhas identificadas + linhas estimadas no lugar das sem apresentação
    ap_col, un_col = pdv_map.get("apresentacao") or "APRESENTACAO", pdv_map["un"]
    pf_col, fa_col = pdv_map.get("pf"), pdv_map.get("fator")
    pdv_ok = pdv_raw.loc[tp.index[~tp["desconhecido"]]].copy()
    pdv_ok["ORIGEM"] = "Identificado"
    est = alocacao.copy()
    est[ap_col] = est["apresentacao_estimada"]
    est[un_col] = est["un_estimada"]
    if pf_col:
        est[pf_col] = est["pf_estimado"].round(2)
    if fa_col:
        est[fa_col] = est["fator_estimado"].round(4)
    est["ORIGEM"] = "Estimado"
    colunas = list(pdv_raw.columns) + [c for c in (ap_col, "ORIGEM") if c not in pdv_raw.columns]
    est = est[[c for c in colunas if c in est.columns]]
    pdv_ok = pdv_ok.reindex(columns=colunas)
    pdv_final = pd.concat([pdv_ok, est], ignore_index=True)

    log = pd.DataFrame(logs)
    resumo = {
        "nivel": cfg.nivel_geo,
        "combinacoes": len(log),
        "exatas": int((log.get("qualidade") == "Exato").sum()) if len(log) else 0,
        "pdv_desconhecidos": int(len(desconhecidos)),
        "un_desconhecidas": float(desconhecidos["un"].sum()),
        "un_alocadas": float(aloc["un_estimada"].sum()),
    }
    return Resultado(alocacao, matriz, log, restantes_saida if restantes_saida is not None else pd.DataFrame(),
                     precos, pdv_final, resumo)
