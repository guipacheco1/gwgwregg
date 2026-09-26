"""Diagnóstico: explica por que as combinações não bateram exatamente.

Olha os dois arquivos já padronizados e responde, com números:
1. Chaves que existem só em um dos arquivos (canal, UF, período, mercado...).
2. Conciliação de unidades por combinação (PDVs sem apresentação x restante do mercado).
3. Unidades já identificadas maiores que o mercado (restante negativo).
4. Preço: o valor/unidade de cada PDV cabe na faixa de preços das apresentações?
   Os dois arquivos estão na mesma base de valor?
5. PDVs repetidos dentro da mesma combinação.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from motor import Config, Resultado, _chaves_base, calcular_restantes, preparar, tabela_precos

ROTULO_CHAVE = {"periodo": "Período", "mercado": "Mercado", "grupo": "Grupo", "canal": "Canal",
                "uf": "UF", "produto": "Produto", "geo": "Geografia"}


@dataclass
class Diagnostico:
    conclusoes: list = field(default_factory=list)    # [(nível, texto)] nível: "erro" | "alerta" | "ok"
    chaves: pd.DataFrame = field(default_factory=pd.DataFrame)
    conciliacao: pd.DataFrame = field(default_factory=pd.DataFrame)
    negativos: pd.DataFrame = field(default_factory=pd.DataFrame)
    precos_pdv: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicados: pd.DataFrame = field(default_factory=pd.DataFrame)
    onde: pd.DataFrame = field(default_factory=pd.DataFrame)          # uma linha por combinação com problema
    resumo_causas: pd.DataFrame = field(default_factory=pd.DataFrame)
    det_pdv: pd.DataFrame = field(default_factory=pd.DataFrame)       # detalhe por PDV (com _chave)
    det_apres: pd.DataFrame = field(default_factory=pd.DataFrame)     # detalhe por apresentação (com _chave)


def _fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def diagnosticar(pdv_raw, pdv_map, merc_raw, merc_map, cfg: Config, res: Resultado | None = None) -> Diagnostico:
    cfg = copy.deepcopy(cfg)
    tp, tm = preparar(pdv_raw, pdv_map, merc_raw, merc_map, cfg)
    base = _chaves_base(tp, tm)
    chave = base + ["geo"]
    d = Diagnostico()
    sem = tp[tp["desconhecido"] & (tp["un"] > 0)]
    un_sem = float(sem["un"].sum())

    # ------------------------------------------------------------ 1. chaves que não casam
    linhas = []
    for c in base + ["uf"]:
        vp = sem.groupby(c)["un"].sum()
        vm = tm.groupby(c)["un"].sum()
        for v, u in vp[~vp.index.isin(vm.index)].items():
            linhas.append({"Campo": ROTULO_CHAVE.get(c, c), "Valor": v, "Existe só no arquivo de": "PDV", "Unidades": u})
        for v, u in vm[~vm.index.isin(vp.index)].items():
            linhas.append({"Campo": ROTULO_CHAVE.get(c, c), "Valor": v, "Existe só no arquivo de": "Mercado", "Unidades": u})
    d.chaves = pd.DataFrame(linhas)
    if len(d.chaves):
        so_pdv = d.chaves[d.chaves["Existe só no arquivo de"] == "PDV"]
        if len(so_pdv):
            u = so_pdv.groupby("Campo")["Unidades"].sum().max()
            campos = ", ".join(sorted(so_pdv["Campo"].unique()))
            d.conclusoes.append(("erro",
                f"Há valores de **{campos}** no arquivo de PDV que não existem no de mercado "
                f"(até {_fmt(u)} unidades, {u / max(un_sem, 1):.0%} do total a quebrar). Esses PDVs ficam sem mercado. "
                "Confira a grafia/código na tabela **Chaves que não casam**."))

    # ------------------------------------------------------------ 2. conciliação de unidades
    rest = calcular_restantes(tp, tm, base, "geo")
    rpos = rest[rest["un_rest"] > 0].groupby(chave, dropna=False)["un_rest"].sum().rename("Mercado restante")
    rneg = rest[rest["un_rest"] < -0.5].groupby(chave, dropna=False)["un_rest"].sum().rename("Restante negativo")
    merc_tot = tm.groupby(chave, dropna=False)["un"].sum().rename("Mercado total")
    ident = tp[~tp["desconhecido"]].groupby(chave, dropna=False)["un"].sum().rename("PDVs identificados")
    pdv_s = sem.groupby(chave, dropna=False)["un"].sum().rename("PDVs a quebrar")
    conc = pd.concat([merc_tot, ident, rpos, rneg, pdv_s], axis=1).fillna(0)
    conc = conc[conc["PDVs a quebrar"] > 0]
    conc["Diferença (PDV − mercado)"] = conc["PDVs a quebrar"] - conc["Mercado restante"]
    conc["Cobertura"] = conc["PDVs a quebrar"] / conc["Mercado restante"].replace(0, np.nan)
    conc["Situação"] = np.select(
        [conc["Mercado restante"] <= 0,
         conc["Diferença (PDV − mercado)"].abs() <= 0.5,
         conc["Diferença (PDV − mercado)"] > 0],
        ["Sem mercado restante", "Bate", "PDVs maiores que o mercado"], "PDVs menores que o mercado")
    conc = conc.reset_index().rename(columns={k: ROTULO_CHAVE.get(k, k) for k in chave})
    d.conciliacao = conc.sort_values("Diferença (PDV − mercado)", key=lambda s: -s.abs())

    n = len(conc)
    if n:
        sit = conc["Situação"].value_counts()
        bate = int(sit.get("Bate", 0))
        maior, menor, semm = (int(sit.get(k, 0)) for k in
                              ("PDVs maiores que o mercado", "PDVs menores que o mercado", "Sem mercado restante"))
        if bate == n:
            d.conclusoes.append(("ok", "As unidades dos PDVs batem com o mercado restante em **todas** as combinações."))
        else:
            com_m = conc[conc["Mercado restante"] > 0]
            cob = com_m["PDVs a quebrar"].sum() / max(com_m["Mercado restante"].sum(), 1)
            d.conclusoes.append(("erro" if bate < 0.5 * n else "alerta",
                f"Unidades batem em só **{bate} de {n}** combinações ({bate / n:.0%}). "
                f"PDVs > mercado: {maior} · PDVs < mercado: {menor} · sem mercado: {semm}. "
                f"Onde há mercado, os PDVs somam **{cob:.0%}** do restante."))
            if menor > maior and cob < 0.98:
                d.conclusoes.append(("alerta",
                    "Como os PDVs somam menos que o mercado, o arquivo de PDV provavelmente não traz todos os "
                    "estabelecimentos (ou o mercado inclui canais/PDVs que não estão no arquivo de PDV)."))
            if maior > menor and cob > 1.02:
                d.conclusoes.append(("alerta",
                    "Como os PDVs somam mais que o mercado, verifique se há PDVs repetidos, se o mercado está "
                    "filtrado (ex.: só alguns produtos/SKUs) ou se o PDV traz produtos que não estão no mercado."))
        # geografia: nível cidade mandando muito para 'outras'
        if cfg.nivel_geo == "cidade":
            outras = sem["geo"].str.contains("OUTRAS", na=False)
            if outras.mean() > 0.5:
                d.conclusoes.append(("alerta",
                    f"No nível Cidade, {outras.mean():.0%} dos PDVs caíram em 'OUTRAS CIDADES'. "
                    "Se o mercado for contratado por UF, rode no nível **UF**."))

    # ------------------------------------------------------------ 3. restante negativo
    neg = rest[rest["un_rest"] < -0.5].copy()
    if len(neg):
        neg = neg.rename(columns={"un_rest": "Restante (negativo)", "apresentacao": "Apresentação"})
        neg = neg[[c for c in chave] + ["Apresentação", "Restante (negativo)"]]
        d.negativos = neg.rename(columns={k: ROTULO_CHAVE.get(k, k) for k in chave})
        d.conclusoes.append(("alerta",
            f"Em {len(neg)} casos os PDVs **já identificados** somam mais que o mercado para a mesma apresentação "
            f"({_fmt(-neg['Restante (negativo)'].sum())} unidades). Isso tira unidades de outras apresentações e "
            "impede fechar exato. Veja **Restante negativo**."))

    # ------------------------------------------------------------ 4. preço
    if cfg.usar_valor and "pf" in pdv_map and pdv_map.get("pf") and len(sem):
        precos = tabela_precos(tm, base)
        chave_preco = [c for c in ("periodo", "mercado", "grupo") if c in base] + ["apresentacao"]
        r = rest[rest["un_rest"] > 0.5].merge(precos, on=chave_preco, how="left")
        faixa = r.groupby(chave, dropna=False)["preco_medio"].agg(preco_min="min", preco_max="max")
        grp = [r[c] for c in chave]
        valor_merc = (r["un_rest"] * r["preco_medio"]).groupby(grp, dropna=False).sum()
        un_merc = r["un_rest"].groupby(grp, dropna=False).sum()
        p = sem.merge(faixa, left_on=chave, right_index=True, how="left")
        p = p[p["pf"] > 0]
        if len(p):
            p["preco_pdv"] = p["pf"] / p["un"]
            tol = 0.01
            p["Situação preço"] = np.select(
                [p["preco_min"].isna(), p["preco_pdv"] < p["preco_min"] * (1 - tol),
                 p["preco_pdv"] > p["preco_max"] * (1 + tol)],
                ["Sem mercado", "Abaixo do menor preço", "Acima do maior preço"], "Dentro da faixa")
            fora = p[p["Situação preço"].isin(["Abaixo do menor preço", "Acima do maior preço"])]
            cols = ["pdv"] + chave + ["un", "pf", "preco_pdv", "preco_min", "preco_max", "Situação preço"]
            d.precos_pdv = (fora[cols].rename(columns={"pdv": "PDV", "un": "Unidades", "pf": "Valor PDV",
                                                       "preco_pdv": "Valor/unidade PDV", "preco_min": "Menor preço mercado",
                                                       "preco_max": "Maior preço mercado", **ROTULO_CHAVE})
                            .sort_values("Unidades", ascending=False))
            if len(fora):
                pct = fora["un"].sum() / p["un"].sum()
                d.conclusoes.append(("erro" if pct > 0.2 else "alerta",
                    f"**{len(fora)} PDVs** ({pct:.0%} das unidades) têm valor/unidade **fora da faixa** de preços das "
                    "apresentações do mercado. Para eles é impossível bater o valor com qualquer combinação."))
            # base de valor: soma PF dos PDVs vs valor do mercado restante
            # compara valor POR UNIDADE (independe de os PDVs cobrirem o mercado todo)
            g = p.groupby(chave, dropna=False)[["pf", "un"]].sum()
            preco_pdv = g["pf"] / g["un"]
            preco_merc = (valor_merc / un_merc).reindex(g.index)
            razao = (preco_pdv / preco_merc).replace([np.inf, -np.inf], np.nan).dropna()
            if len(razao):
                med = float(razao.median())
                if abs(med - 1) > 0.05:
                    d.conclusoes.append(("erro",
                        f"O valor por unidade dos PDVs é, em média (mediana), **{med:.2f}×** o do mercado."
                        " Os arquivos parecem estar em **bases de valor diferentes** (ex.: PF × PMC, com/sem "
                        "desconto, R$ × R$ mil). Nesse caso, desmarque **Valor (PF)** ou corrija a base."))
                elif len(fora) == 0:
                    d.conclusoes.append(("ok", "Os valores dos PDVs estão na mesma base e dentro da faixa de preços do mercado."))

    # ------------------------------------------------------------ 5. PDVs repetidos
    if "pdv" in sem:
        dup = sem.groupby(chave + ["pdv"], dropna=False).size()
        dup = dup[dup > 1]
        if len(dup):
            d.duplicados = dup.rename("Linhas").reset_index().rename(columns={"pdv": "PDV", **ROTULO_CHAVE})
            d.conclusoes.append(("alerta",
                f"**{len(dup)} PDVs** aparecem em mais de uma linha na mesma combinação. Cada linha é quebrada "
                "separadamente; se forem duplicadas, as unidades estão contadas em dobro."))

    # ------------------------------------------------------------ motivos do resultado
    if res is not None and len(res.log) and "motivo" in res.log:
        mot = res.log.loc[res.log["qualidade"] != "Exato", "motivo"].fillna("")
        so_valor = (mot == "Valor (PF) fora da tolerância").sum()
        if so_valor and so_valor >= 0.5 * max(len(mot), 1):
            d.conclusoes.append(("alerta",
                f"Em {so_valor} combinações as unidades fecham e **só o valor** não bate. Se o valor/unidade dos PDVs "
                "estiver dentro da faixa, a causa é o preço variar entre PDVs para a mesma apresentação; "
                "a quebra de unidades continua correta."))
    if res is not None and len(res.log):
        _onde_esta_o_erro(d, tp, tm, base, res, cfg, pdv_map)
    if not d.conclusoes:
        d.conclusoes.append(("ok", "Não encontrei inconsistências entre os arquivos."))
    ordem = {"erro": 0, "alerta": 1, "ok": 2}
    d.conclusoes.sort(key=lambda x: ordem[x[0]])
    return d


# =============================================================================
# "Onde está o erro": causa exata por combinação
# =============================================================================
def _pct_br(x: float) -> str:
    return f"{x:+.1%}".replace(".", ",")


def _brl(x: float) -> str:
    return "R$ " + f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _texto_chave(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + " | " + df[c].astype(str)
    return out


def _campo_sem_mercado(linha: dict, chave: list[str], tm: pd.DataFrame, geo_col: str) -> tuple[str, str]:
    """Descobre qual campo da combinação não existe no arquivo de mercado."""
    tm2 = tm.assign(_g=tm[geo_col])
    campos = [c for c in chave if c != geo_col] + [geo_col]
    for c in campos:
        col = "_g" if c == geo_col else c
        outros = [x for x in campos if x != c]
        m = pd.Series(True, index=tm2.index)
        for o in outros:
            m &= tm2["_g" if o == geo_col else o] == linha[o]
        if m.any() and not (tm2.loc[m, col] == linha[c]).any():
            return ROTULO_CHAVE.get("geo" if c == geo_col else c, c), str(linha[c])
        if not (tm2[col] == linha[c]).any():
            return ROTULO_CHAVE.get("geo" if c == geo_col else c, c), str(linha[c])
    return "", ""


def _onde_esta_o_erro(d: Diagnostico, tp, tm, base, res, cfg, pdv_map):
    log = res.log
    if "_chave" not in log:
        return
    sem = tp[tp["desconhecido"] & (tp["un"] > 0)].copy()
    precos = tabela_precos(tm, base)
    chave_preco = [c for c in ("periodo", "mercado", "grupo") if c in base] + ["apresentacao"]
    alvo = log[log["qualidade"] != "Exato"]

    # restantes por nível geográfico usado no log
    rests = {}
    for gcol in alvo["_geo_col"].dropna().unique():
        r = calcular_restantes(tp, tm, base, gcol).merge(precos, on=chave_preco, how="left")
        r["preco_medio"] = r["preco_medio"].fillna(0)
        r["fator_un"] = r["fator_un"].fillna(0)
        r["_chave"] = _texto_chave(r, base + [gcol])
        rests[gcol] = r
        sem["_chave_" + gcol] = _texto_chave(sem, base + [gcol])

    aloc = res.alocacao
    col_pdv = pdv_map.get("pdv")
    est = pd.DataFrame(columns=["_chave", "pdv", "un_est", "pf_est", "fator_est"])
    if len(aloc) and col_pdv in aloc:
        est = (aloc.assign(pdv=aloc[col_pdv].astype(str).str.strip())
               .groupby(["_chave", "pdv"])[["un_estimada", "pf_estimado", "fator_estimado"]].sum()
               .rename(columns={"un_estimada": "un_est", "pf_estimado": "pf_est", "fator_estimado": "fator_est"})
               .reset_index())

    por_rest = {g: dict(tuple(r.groupby("_chave"))) for g, r in rests.items()}
    por_sem = {g: dict(tuple(sem.groupby("_chave_" + g))) for g in rests}
    por_est = dict(tuple(est.groupby("_chave"))) if len(est) else {}
    por_aloc = dict(tuple(aloc.groupby("_chave"))) if len(aloc) and "_chave" in aloc else {}
    vazio_r = next(iter(rests.values())).iloc[0:0] if rests else pd.DataFrame()
    linhas, det_pdv, det_ap = [], [], []
    for _, lg in alvo.iterrows():
        gcol = lg["_geo_col"]
        k = lg["_chave"]
        chave = base + [gcol]
        rot_geo = "UF" if gcol == "geo_uf" else {"uf": "UF", "cidade": "UF | Cidade", "regiao": "UF | Região"}.get(cfg.nivel_geo, "UF")
        rotulo = {(rot_geo if c == gcol else ROTULO_CHAVE.get(c, c)): lg.get(c) for c in chave}
        rk = por_rest[gcol].get(k, vazio_r)
        pk = por_sem[gcol].get(k, sem.iloc[0:0])
        un_pdv = float(pk["un"].sum())
        pos = rk[rk["un_rest"] > 0.5]
        neg = rk[rk["un_rest"] < -0.5]
        un_merc = float(pos["un_rest"].sum())
        val_pdv = float(pk["pf"].sum())
        val_merc = float((pos["un_rest"] * pos["preco_medio"]).sum())
        fat_pdv = float(pk["fator"].sum())
        fat_merc = float((pos["un_rest"] * pos["fator_un"]).sum())
        pmin = pos["preco_medio"].min() if len(pos) else np.nan
        pmax = pos["preco_medio"].max() if len(pos) else np.nan

        # detalhe por PDV
        g = pk.groupby("pdv")[["un", "pf", "fator"]].sum().reset_index()
        ek = por_est.get(k, est.iloc[0:0]).drop(columns="_chave")
        g = g.merge(ek, on="pdv", how="left").fillna(0)
        g["preco_un"] = g["pf"] / g["un"].replace(0, np.nan)
        g["fora_faixa"] = (g["pf"] > 0) & ((g["preco_un"] < pmin * 0.99) | (g["preco_un"] > pmax * 1.01))
        g["dif_valor_pct"] = (g["pf_est"] - g["pf"]) / g["pf"].replace(0, np.nan)
        g["repetido"] = g["pdv"].isin(pk["pdv"][pk["pdv"].duplicated()])
        g["_chave"] = k
        det_pdv.append(g)
        # detalhe por apresentação
        ap = rk[["apresentacao", "un_rest", "preco_medio", "fator_un"]].copy()
        ak = por_aloc.get(k)
        if ak is not None:
            al = ak.groupby("apresentacao_estimada")["un_estimada"].sum()
            ap["un_alocada"] = ap["apresentacao"].map(al).fillna(0)
        else:
            ap["un_alocada"] = 0.0
        ap["_chave"] = k
        det_ap.append(ap)

        causa, detalhe, acao = "", "", ""
        tol_v = max(0.01 * val_merc, 1.0)
        if lg["qualidade"] == "Não alocado":
            campo, valor = _campo_sem_mercado(lg.to_dict(), chave, tm, gcol)
            causa = f"{campo or 'Chave'} não existe no arquivo de mercado"
            detalhe = (f"{campo} '{valor}' dos PDVs não aparece no arquivo de mercado"
                       if campo else "A combinação dos PDVs não existe no arquivo de mercado")
            detalhe += f" ({_fmt(un_pdv)} un. sem mercado)."
            acao = f"Corrija/padronize o valor de {campo or 'chave'} em um dos arquivos."
        elif len(neg):
            top = neg.sort_values("un_rest").iloc[0]
            causa = "PDVs identificados passam do mercado"
            detalhe = (f"Em '{top['apresentacao']}' os PDVs já identificados têm {_fmt(-top['un_rest'])} un. a mais "
                       f"que o mercado ({len(neg)} apresentação(ões) nessa situação).")
            acao = "Confira a apresentação informada nesses PDVs ou o volume do mercado para essa combinação."
        elif abs(un_pdv - un_merc) > 0.5:
            dif = un_pdv - un_merc
            rep_ = g[g["repetido"]]
            if dif > 0 and len(rep_):
                causa = "PDV repetido"
                detalhe = (f"PDVs somam {_fmt(un_pdv)} un. × mercado {_fmt(un_merc)} un. (+{_fmt(dif)}). "
                           f"PDV(s) em mais de uma linha: {', '.join(rep_['pdv'].astype(str).head(5))}.")
                acao = "Remova linhas duplicadas do arquivo de PDV."
            elif dif > 0:
                causa = "PDVs têm mais unidades que o mercado"
                detalhe = f"PDVs somam {_fmt(un_pdv)} un. × mercado restante {_fmt(un_merc)} un. (+{_fmt(dif)} un.)."
                acao = ("Confira se o mercado está completo para essa combinação (todas as apresentações/canais) "
                        "ou se há PDVs classificados no canal/UF errado.")
            else:
                causa = "Mercado tem mais unidades que os PDVs"
                detalhe = (f"Mercado restante {_fmt(un_merc)} un. × PDVs {_fmt(un_pdv)} un. "
                           f"(faltam {_fmt(-dif)} un., cobertura {un_pdv / max(un_merc, 1):.0%}).")
                acao = "Faltam PDVs (ou unidades) no arquivo de PDV para essa combinação."
        elif cfg.usar_valor and "Valor" in str(lg.get("motivo", "")):
            fora = g[g["fora_faixa"]]
            if abs(val_pdv - val_merc) > tol_v:
                causa = "Valor total dos PDVs ≠ valor do mercado"
                detalhe = (f"Mesmas {_fmt(un_pdv)} un., mas PDVs somam {_brl(val_pdv)} × mercado {_brl(val_merc)} "
                           f"({_pct_br(val_pdv / max(val_merc, 1e-9) - 1)}). Com valores totais diferentes é impossível fechar.")
                acao = "Os arquivos estão em bases de valor diferentes ou o preço mudou: corrija a base ou desmarque Valor (PF)."
                if len(fora):
                    top = fora.sort_values("un", ascending=False).head(5)
                    detalhe += (" Culpados: PDV " + ", ".join(f"{x.pdv} ({_brl(x.preco_un)}/un.)" for x in top.itertuples())
                                + f", com preço fora da faixa das apresentações ({_brl(pmin)} a {_brl(pmax)}).")
                    acao = "Corrija unidades/valor desses PDVs no arquivo de PDV."
                    causa = "PDV com preço impossível"
            elif len(fora):
                f0 = fora.sort_values("un", ascending=False).iloc[0]
                causa = "PDV com preço impossível"
                detalhe = (f"PDV {f0['pdv']}: {_brl(f0['preco_un'])}/un., fora da faixa das apresentações "
                           f"({_brl(pmin)} a {_brl(pmax)}). {len(fora)} PDV(s) nessa situação.")
                acao = "Confira unidades/valor desses PDVs no arquivo de PDV."
            elif lg.get("status") == "Limite de tempo":
                causa = "Tempo esgotado"
                detalhe = "O otimizador não terminou dentro do tempo máximo para essa combinação."
                acao = "Aumente o 'Tempo máx. por combinação' em Avançado."
            else:
                pior = g.reindex(g["dif_valor_pct"].abs().sort_values(ascending=False).index).head(3)
                lista = "; ".join(f"PDV {x.pdv}: real {_brl(x.pf)} × estimado {_brl(x.pf_est)}" for x in pior.itertuples())
                causa = "Não existe combinação exata para o valor"
                detalhe = ("Unidades e valor total batem, mas o valor de cada PDV não é uma soma exata de "
                           f"apresentações × preço médio (preço varia entre PDVs). Mais distantes: {lista}.")
                acao = "A quebra de unidades está correta; a diferença é de preço por PDV. Pode aceitar ou desmarcar Valor (PF)."
        elif cfg.usar_primo and "primo" in str(lg.get("motivo", "")):
            if abs(fat_pdv - fat_merc) > 0.011:
                causa = "Fator total dos PDVs ≠ fator do mercado"
                detalhe = f"PDVs somam fator {fat_pdv:.2f} × mercado {fat_merc:.2f}."
                acao = "Confira a coluna de fator (nº primo) nos dois arquivos."
            else:
                causa = "Não existe combinação exata para o número primo"
                detalhe = "O fator total bate, mas o de algum PDV não fecha com nenhuma soma de apresentações."
                acao = "Confira o fator dos PDVs com maior diferença (detalhe abaixo)."
        else:
            causa = "Mix de apresentações não fecha"
            detalhe = (f"Unidades batem ({_fmt(un_pdv)}), mas a distribuição por apresentação ficou "
                       f"{_fmt(lg.get('desvio_un_mercado', 0))} un. diferente do mercado (troca feita para bater o valor).")
            acao = "Aumente o peso 'bater total do mercado' ou desmarque Valor (PF)."

        linhas.append({**rotulo, "Causa": causa, "Detalhe": detalhe, "Como resolver": acao,
                       "Unidades PDV": un_pdv, "Unidades mercado": un_merc, "_chave": k})

    d.onde = pd.DataFrame(linhas)
    if len(d.onde):
        d.resumo_causas = (d.onde.groupby("Causa").agg(Combinações=("Causa", "size"),
                                                       Unidades_PDV=("Unidades PDV", "sum"))
                           .sort_values("Combinações", ascending=False).reset_index()
                           .rename(columns={"Unidades_PDV": "Unidades PDV envolvidas"}))
        d.resumo_causas["% das combinações"] = (d.resumo_causas["Combinações"] / len(log)).map("{:.0%}".format)
    d.det_pdv = pd.concat(det_pdv, ignore_index=True) if det_pdv else pd.DataFrame()
    d.det_apres = pd.concat(det_ap, ignore_index=True) if det_ap else pd.DataFrame()


# =============================================================================
# Explicação em linguagem simples
# =============================================================================
_SIMPLES = [
    # (trecho da causa, emoji, explicação simples, o que fazer)
    ("Mercado tem mais unidades", "📦",
     "**Sobrou mercado.** O arquivo de mercado diz que vendeu mais do que os PDVs do seu arquivo compraram. "
     "É como dividir 100 balas entre crianças que só pediram 70: sobram 30 balas e a conta não fecha.",
     "Normalmente faltam PDVs no arquivo de PDV (ou parte da venda foi para quem não está no arquivo)."),
    ("PDVs têm mais unidades", "📦",
     "**Faltou mercado.** Os PDVs compraram mais do que o mercado diz que vendeu. "
     "É como 10 crianças pedindo 120 balas quando só existem 100.",
     "Confira se o arquivo de mercado está completo ou se há PDV no canal/UF errado."),
    ("PDV repetido", "👯",
     "**PDV em dobro.** O mesmo PDV aparece em mais de uma linha, então as unidades dele contam duas vezes.",
     "Tire as linhas repetidas do arquivo de PDV."),
    ("não existe no arquivo de mercado", "🔍",
     "**Nome diferente.** Um canal, UF, período ou mercado está escrito de um jeito num arquivo e de outro no outro "
     "(ex.: “GOV” e “Governo”). O app não acha o par e esses PDVs ficam sem mercado.",
     "Deixe o nome igual nos dois arquivos."),
    ("identificados passam do mercado", "📦",
     "**Já passou do total.** Os PDVs que já têm apresentação compraram mais daquela apresentação do que o mercado inteiro vendeu.",
     "Confira a apresentação desses PDVs ou o volume do mercado."),
    ("Valor total dos PDVs", "💰",
     "**Os preços dos dois arquivos são diferentes.** As quantidades batem, mas o dinheiro não. "
     "É como a nota do mercado dizer que o lanche custa R$ 100 e a sua dizer R$ 125: nunca vai fechar.",
     "Use arquivos na mesma base de preço, ou desmarque “Valor (PF)” para quebrar só por unidades."),
    ("preço impossível", "💰",
     "**Preço que não existe.** Tem PDV pagando por unidade mais caro que a apresentação mais cara "
     "(ou mais barato que a mais barata). Nenhuma divisão consegue chegar nesse valor.",
     "Confira unidades e valor desses PDVs."),
    ("combinação exata para o valor", "🧩",
     "**Quase fechou.** Quantidade e dinheiro total batem, mas cada PDV pagou um preço um pouquinho diferente, "
     "então não dá para deixar o valor de **todos** perfeito ao mesmo tempo. A quebra de unidades está certa.",
     "Pode aceitar o resultado. Se quiser 100%, desmarque “Valor (PF)”."),
    ("Tempo esgotado", "⏱️",
     "**Não deu tempo.** A conta era grande e o app parou antes de terminar.",
     "Aumente o “Tempo máx. por combinação” em Avançado."),
    ("Fator total", "🔢",
     "**O número primo dos arquivos não bate.** A soma do fator dos PDVs é diferente da do mercado.",
     "Confira a coluna de fator nos dois arquivos, ou desmarque “Número primo”."),
    ("combinação exata para o número primo", "🔢",
     "**Quase fechou no número primo.** O total bate, mas o de algum PDV não fecha com nenhuma soma de apresentações.",
     "Confira o fator dos PDVs marcados no detalhe."),
    ("Mix de apresentações", "🧩",
     "**Trocou apresentação para acertar o valor.** As unidades batem, mas para acertar o valor de cada PDV "
     "o app teve que tirar um pouco de uma apresentação e pôr em outra.",
     "Se preferir bater o mercado, aumente o peso “bater total do mercado” ou desmarque “Valor (PF)”."),
]

_CULPA_ARQUIVO = ("Mercado tem mais", "PDVs têm mais", "PDV repetido", "não existe no arquivo", "identificados passam",
                  "Valor total dos PDVs", "preço impossível", "Fator total")


def explicar_simples(d: Diagnostico, res) -> str:
    r = res.resumo
    n, e = r["combinacoes"], r["exatas"]
    if not n:
        return "Nenhuma combinação foi processada. Confira os filtros."
    pct = e / n
    txt = [f"### {'✅' if e == n else '📊'} {e} de {n} combinações fecharam certinho ({pct:.0%})",
           "_Combinação = um pedaço do mercado: o mesmo mês + o mesmo mercado + o mesmo canal + a mesma UF (ou cidade). "
           "Em cada uma, o app divide o que o mercado vendeu entre os PDVs. **Fechar certinho** = a divisão bateu "
           "exatamente as unidades, o valor e (se marcado) o número primo._"]
    if e == n:
        txt.append("Tudo fechou. Os dois arquivos conversam perfeitamente. 🎉")
        return "\n\n".join(txt)
    faltam = n - e
    txt.append(f"**{'A outra não fechou' if faltam == 1 else f'As outras {faltam} não fecharam'} "
               f"{'por este motivo' if len(d.resumo_causas) == 1 else 'por estes motivos'}:**")
    rc = d.resumo_causas if len(d.resumo_causas) else pd.DataFrame(columns=["Causa", "Combinações"])
    culpa_arquivo = 0
    for _, x in rc.iterrows():
        info = next((t for t in _SIMPLES if t[0] in x["Causa"]), None)
        emoji, simples, fazer = (info[1], info[2], info[3]) if info else ("•", x["Causa"], "")
        qtd = f"{x['Combinações']} {'combinação' if x['Combinações'] == 1 else 'combinações'}"
        txt.append(f"{emoji} **{qtd} ({x['Combinações'] / n:.0%})** — {simples}  \n"
                   f"👉 *O que fazer:* {fazer}")
        if any(c in x["Causa"] for c in _CULPA_ARQUIVO):
            culpa_arquivo += x["Combinações"]
    if culpa_arquivo >= 0.5 * (n - e):
        txt.append("**Resumindo:** o app fez a conta certa; o que não fecha são os **números dos arquivos** "
                   "(eles não conversam entre si). Corrigindo o que está acima, a porcentagem sobe.")
    else:
        txt.append("**Resumindo:** os arquivos conversam bem; a diferença vem de **preço variando entre PDVs**. "
                   "A quebra de unidades está correta.")
    txt.append("Para ver exatamente **onde**, use a tabela **Onde está o erro** e o **Abrir uma combinação** abaixo.")
    return "\n\n".join(txt)
