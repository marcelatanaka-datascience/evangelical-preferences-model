#!/usr/bin/env python3
"""
Análise CESOP Completa — Todas as Pesquisas com Religião
=========================================================
Produz ilustrações quantitativas para a tese de doutorado sobre
preferências políticas de eleitores religiosos no Brasil.

Combina:
1. Análises já mapeadas em cesop_config.csv (usando cesop_relatorio_religiao.txt)
2. Análises novas das pesquisas RNSP/IPEC 2020-2022 com RELIGIAO codificada:
   1=Católico, 2-11=Evangélico, 12=Adventista, 19=Sem religião, 20=Ateu

Output: cesop_ilustracoes_tese.md
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/home/user/evangelical-preferences-model/database_extractions/cesop-researches/csv')
OUT = Path('/home/user/evangelical-preferences-model/database_extractions/cesop-researches/sampled_questionnaires')

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def recode_religion_new(df, col, evang_codes=None, cat_codes=None):
    """Recode for IPEC 2020-2022 surveys: 1=Catholic, 2-11=Evangelical."""
    if evang_codes is None:
        evang_codes = list(range(2, 12))  # 2–11
    if cat_codes is None:
        cat_codes = [1]
    evang_set = set(evang_codes)
    cat_set = set(cat_codes)
    def classify(v):
        if pd.isna(v): return 'NR/NS'
        try:
            vi = int(float(v))
        except:
            return 'NR/NS'
        if vi in evang_set: return 'Evangélico'
        if vi in cat_set: return 'Católico'
        return 'Outro'
    return df[col].apply(classify)


def crosstab_analysis(df, var, relig_series, val_labels=None, skip_vals=None):
    """Run crosstab + chi-square for one variable against religion series."""
    if var not in df.columns:
        return None
    skip = set(skip_vals or [99, 98, 97, 96])
    data = df[[var]].copy()
    data['relig'] = relig_series
    data = data[data[var].notna()].copy()
    # Remove skip values
    try:
        data = data[~data[var].apply(lambda x: int(float(x)) if not pd.isna(x) else -1).isin(skip)]
    except:
        pass
    if len(data) < 30:
        return None

    ct_pct = pd.crosstab(data[var], data['relig'], normalize='columns') * 100
    ct_counts = pd.crosstab(data[var], data['relig'], margins=True)

    # Chi-square: Evangélico vs Católico only
    mask = data['relig'].isin(['Evangélico', 'Católico'])
    ct_chi = pd.crosstab(data[mask][var], data[mask]['relig'])
    chi2, p_val, dof = None, None, None
    if ct_chi.shape[0] > 1 and ct_chi.shape[1] > 1:
        try:
            chi2, p_val, dof, _ = chi2_contingency(ct_chi)
        except:
            pass

    n_evang = (relig_series == 'Evangélico').sum()
    n_cat = (relig_series == 'Católico').sum()
    n_outro = (relig_series == 'Outro').sum()

    return {
        'var': var,
        'val_labels': val_labels or {},
        'n_evang': n_evang,
        'n_cat': n_cat,
        'n_outro': n_outro,
        'n_total': len(relig_series),
        'chi2': chi2,
        'p_value': p_val,
        'dof': dof,
        'ct_pct': ct_pct,
        'ct_counts': ct_counts,
    }


def sig_stars(p):
    if p is None: return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


def format_result_md(r, title, arquivo, ano, note=''):
    """Format a single crosstab result as Markdown."""
    lines = []
    if r is None:
        return ''

    chi_str = ''
    if r['chi2'] is not None:
        chi_str = f" | χ²={r['chi2']:.1f}, p={r['p_value']:.4f} {sig_stars(r['p_value'])}, gl={r['dof']}"

    lines.append(f"\n**{title}**")
    lines.append(f"*Arquivo: {arquivo} | Ano: {ano} | N={r['n_total']} (Ev={r['n_evang']}, Cat={r['n_cat']}){chi_str}*")
    if note:
        lines.append(f"*{note}*")

    ct = r['ct_pct']
    vl = r['val_labels']

    # Build table
    cols_present = [c for c in ['Evangélico', 'Católico', 'Outro'] if c in ct.columns]
    if not cols_present:
        return ''

    header = '| Resposta | ' + ' | '.join(cols_present) + ' |'
    sep    = '|' + '---|' * (len(cols_present) + 1)
    lines.append(header)
    lines.append(sep)

    for val in ct.index:
        label = vl.get(val, vl.get(float(val) if not isinstance(val, str) else val, str(val)))
        row_vals = []
        for c in cols_present:
            if c in ct.columns:
                row_vals.append(f"{ct.loc[val, c]:.1f}%")
            else:
                row_vals.append('-')
        lines.append(f"| {label} | " + ' | '.join(row_vals) + ' |')

    # Highlight biggest evangelical vs catholic difference
    if 'Evangélico' in ct.columns and 'Católico' in ct.columns:
        diffs = {}
        for v in ct.index:
            diffs[v] = abs(ct.loc[v, 'Evangélico'] - ct.loc[v, 'Católico'])
        if diffs:
            mv = max(diffs, key=diffs.get)
            ml = vl.get(mv, vl.get(float(mv) if not isinstance(mv, str) else mv, str(mv)))
            ev = ct.loc[mv, 'Evangélico']
            ca = ct.loc[mv, 'Católico']
            lines.append(f"\n> **Maior diferença:** *{ml}* → Evangélicos={ev:.1f}% vs Católicos={ca:.1f}% (Δ={abs(ev-ca):.1f}pp)")

    return '\n'.join(lines)


# ─────────────────────────────────────────────
# Load pesquisas metadata
# ─────────────────────────────────────────────
try:
    meta_df = pd.read_csv(OUT.parent / 'pesquisas_cesop_com_religiao.csv', encoding='utf-8-sig')
    meta_map = {}
    for _, row in meta_df.iterrows():
        num = str(row.get('Número', '')).zfill(5)
        meta_map[num] = {
            'titulo': str(row.get('Título', '')),
            'data': str(row.get('Data', '')),
            'local': str(row.get('Local', '')),
            'instituto': str(row.get('Instituto', '')),
        }
except:
    meta_map = {}

def get_meta(arquivo_num):
    num = str(arquivo_num).zfill(5)
    return meta_map.get(num, {})


# ─────────────────────────────────────────────
# Parse existing cesop_relatorio_religiao.txt
# ─────────────────────────────────────────────
def parse_existing_report(report_path):
    """Parse the existing text report into structured results."""
    results = []
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.split('\n' + '-' * 80)
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        if not lines:
            continue
        r = {}
        for line in lines:
            if line.startswith('ARQUIVO:'):
                r['arquivo'] = line.replace('ARQUIVO:', '').strip()
            elif line.startswith('TEMA:'):
                r['tema'] = line.replace('TEMA:', '').strip()
            elif line.startswith('VARIÁVEL:'):
                r['variavel'] = line.replace('VARIÁVEL:', '').strip()
            elif line.startswith('N total='):
                parts = line.split('|')
                for p in parts:
                    p = p.strip()
                    if p.startswith('N total='):
                        try: r['n_total'] = int(p.split('=')[1])
                        except: pass
                    elif p.startswith('Evangélicos='):
                        try: r['n_evang'] = int(p.split('=')[1])
                        except: pass
                    elif p.startswith('Católicos='):
                        try: r['n_cat'] = int(p.split('=')[1])
                        except: pass
            elif line.startswith('Qui-quadrado:'):
                r['chi2_str'] = line.replace('Qui-quadrado:', '').strip()
                # Extract p-value
                try:
                    p_part = [x for x in line.split(',') if 'p=' in x][0]
                    p_val = float(p_part.strip().split('=')[1].split(' ')[0])
                    r['p_value'] = p_val
                    if '***' in line: r['sig'] = '***'
                    elif '**' in line: r['sig'] = '**'
                    elif '*' in line and '***' not in line and '**' not in line: r['sig'] = '*'
                    else: r['sig'] = 'n.s.'
                except:
                    pass
            elif 'DADO PRINCIPAL PARA O TEXTO:' in line:
                pass
            elif line.strip().startswith('Maior diferença:') or 'Maior diferença:' in line:
                r['dado_principal'] = line.strip()
        if r.get('tema'):
            results.append(r)
    return results


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────

sections = {
    'valores_morais': [],
    'pratica_religiosa': [],
    'identidade_publica': [],
    'confianca_institucional': [],
    'cultura_politica': [],
    'eleicoes_democracia': [],
    'lgbtqia_genero': [],
    'raca_diversidade': [],
}

# Keywords to classify into themes
def classify_theme(label):
    label_lower = label.lower()
    if any(k in label_lower for k in ['aborto', 'virgem', 'casamento', 'família', 'familiar', 'sexual', 'moral', 'norma', 'valores famil']):
        return 'valores_morais'
    if any(k in label_lower for k in ['frequência', 'missa', 'culto', 'cerimônia', 'religiosidade', 'grupo religioso', 'membro', 'prática', 'importância da religião']):
        return 'pratica_religiosa'
    if any(k in label_lower for k in ['declarar religião', 'à vontade', 'tv e rádio', 'preconceito por causa', 'tratamento da religião', 'tolerância', 'identidade']):
        return 'identidade_publica'
    if any(k in label_lower for k in ['confiança', 'confiança na igreja', 'confiança no governo', 'confiança nas igrejas', 'avaliação da atuação', 'ics', 'cara da democracia']):
        return 'confianca_institucional'
    if any(k in label_lower for k in ['democracia', 'ditadura', 'partidos', 'governo', 'político', 'política', 'corrupção', 'esquerda', 'direita', 'ideológico', 'polarização']):
        return 'cultura_politica'
    if any(k in label_lower for k in ['eleição', 'bolsonaro', 'haddad', 'lula', 'voto', '2018', '2021', 'presidente', 'arrependido']):
        return 'eleicoes_democracia'
    if any(k in label_lower for k in ['lgbtqia', 'lgbt', 'diversidade sexual', 'mulher', 'gênero', 'orient']):
        return 'lgbtqia_genero'
    if any(k in label_lower for k in ['negro', 'negra', 'racismo', 'preconceito racial', 'raca']):
        return 'raca_diversidade'
    return 'cultura_politica'  # fallback


# ─── 1. Parse existing report ───
report_path = OUT / 'cesop_relatorio_religiao.txt'
if report_path.exists():
    existing = parse_existing_report(report_path)
    print(f"Carregadas {len(existing)} análises do relatório existente")
else:
    existing = []
    print("AVISO: cesop_relatorio_religiao.txt não encontrado")


# ─── 2. NEW: 04748 — Democracia e Eleições (IPEC, Set 2021, N=1009) ───
print("\n[04748] Democracia e Eleições 2021...")
df48 = pd.read_csv(BASE / '04748.csv', encoding='utf-8')
relig48 = recode_religion_new(df48, 'RELIGIAO')

results_04748 = []

# EL01: Intenção de voto Bolsonaro (2021)
vl_el01 = {1: 'Votaria em Bolsonaro', 2: 'Votaria em outro qualquer',
            3: 'Votaria em outro, desde que não Lula', 4: 'Nenhum/Branco/Nulo'}
r = crosstab_analysis(df48, 'EL01', relig48, val_labels=vl_el01, skip_vals=[98, 99])
if r:
    r['tema'] = 'eleicoes_democracia'
    r['label'] = 'Se a eleição p/ Presidente fosse hoje, você votaria em Bolsonaro?'
    r['arquivo'] = '04748'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Set/2021, N=1.009, nacional. Simulação de 2º turno com Bolsonaro.'
    results_04748.append(r)

# VP01: Voto no 2º turno de 2018
vl_vp01 = {1: 'Fernando Haddad (PT)', 2: 'Jair Bolsonaro (PSL)',
            3: 'Branco/Nulo', 4: 'Não votou'}
r = crosstab_analysis(df48, 'VP01', relig48, val_labels=vl_vp01, skip_vals=[98, 99])
if r:
    r['tema'] = 'eleicoes_democracia'
    r['label'] = 'Em quem você votou no 2º turno das eleições de 2018?'
    r['arquivo'] = '04748'
    r['ano'] = '2021'
    r['nota'] = 'Voto retrospectivo 2018.'
    results_04748.append(r)

# VP03: Arrependimento Bolsonaro
vl_vp03 = {1: 'Está arrependido(a)', 2: 'Não está arrependido(a)'}
r = crosstab_analysis(df48, 'VP03', relig48, val_labels=vl_vp03, skip_vals=[98, 99])
if r:
    r['tema'] = 'eleicoes_democracia'
    r['label'] = 'Você está arrependido(a) de ter votado em Bolsonaro em 2018?'
    r['arquivo'] = '04748'
    r['ano'] = '2021'
    r['nota'] = 'Perguntado apenas a quem declarou ter votado em Bolsonaro em 2018.'
    results_04748.append(r)

# CONF01A: Confiança no Governo Federal
vl_conf = {1: 'Confia muito', 2: 'Confia pouco', 3: 'Não confia', 99: 'Sem opinião'}
for var, inst in [('CONF01A', 'Governo Federal'), ('CONF01B', 'Congresso Nacional'),
                   ('CONF01C', 'STF'), ('CONF01D', 'TSE'), ('CONF01E', 'Partidos Políticos')]:
    r = crosstab_analysis(df48, var, relig48, val_labels=vl_conf, skip_vals=[98, 99])
    if r:
        r['tema'] = 'confianca_institucional'
        r['label'] = f'Confiança no(a) {inst}'
        r['arquivo'] = '04748'
        r['ano'] = '2021'
        r['nota'] = f'IPEC, Set/2021. Escala: Confia muito / Confia pouco / Não confia.'
        results_04748.append(r)

# STF01: Opinião sobre o STF
vl_stf01 = {1: 'Deve ser preservado como está',
            2: 'Deve ser reformado, mas mantido',
            3: 'Deve ser fechado/prejudica o Brasil',
            4: 'Não sabe'}
r = crosstab_analysis(df48, 'STF01', relig48, val_labels=vl_stf01, skip_vals=[98, 99])
if r:
    r['tema'] = 'cultura_politica'
    r['label'] = 'Opinião sobre o Supremo Tribunal Federal (STF)'
    r['arquivo'] = '04748'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Set/2021.'
    results_04748.append(r)

# CC01A-F: Polarização política
vl_cc = {1: 'Concorda totalmente', 2: 'Concorda em parte',
          3: 'Discorda em parte', 4: 'Discorda totalmente'}
cc_labels = {
    'CC01A': 'O debate político nas redes sociais é agressivo e intolerante',
    'CC01B': 'O debate político no país em geral é agressivo e intolerante',
    'CC01C': 'Não discuto política pois temo ser cancelado/tratado agressivamente',
    'CC01D': 'A divisão entre esquerda e direita na política não faz sentido',
    'CC01E': 'As pessoas de esquerda são intolerantes com quem pensa diferente',
    'CC01F': 'As pessoas de direita são intolerantes com quem pensa diferente',
}
for var, label in cc_labels.items():
    r = crosstab_analysis(df48, var, relig48, val_labels=vl_cc, skip_vals=[98, 99])
    if r:
        r['tema'] = 'cultura_politica'
        r['label'] = label
        r['arquivo'] = '04748'
        r['ano'] = '2021'
        r['nota'] = 'IPEC, Set/2021. Polarização e debate político.'
        results_04748.append(r)

# VA01: Valores sociais prioritários (top 3)
# Codificado como múltiplas colunas VA01_1, VA01_2, VA01_3
va01_labels = {
    1: 'Amazônia e meio ambiente', 2: 'Combate à pobreza e à fome',
    3: 'Mudanças climáticas', 4: 'Combate ao preconceito',
    5: 'Combate à corrupção', 6: 'Defesa dos direitos LGBTQIA+',
    7: 'Direitos das mulheres', 8: 'Direitos da população negra',
    9: 'Economia forte e empregos', 10: 'Educação gratuita e de qualidade',
    11: 'Saúde gratuita e de qualidade', 12: 'Preservação dos valores da família brasileira',
    13: 'Sociedade livre de violência', 14: 'Combate às desigualdades sociais',
    15: 'Liberdade de expressão',
}

# Criar dummy para cada valor — especialmente 12 (valores familiares) e 6 (LGBTQIA+)
df48_va = df48.copy()
df48_va['relig'] = relig48

for val_code, val_label in va01_labels.items():
    # 1 se mencionado em qualquer das 3 posições
    df48_va[f'va_{val_code}'] = df48_va.apply(
        lambda row: 1 if any(
            str(int(float(row[f'VA01_{i}']))) == str(val_code)
            for i in [1,2,3]
            if pd.notna(row.get(f'VA01_{i}')) and str(row.get(f'VA01_{i}')) not in ['99','98']
        ) else 0, axis=1
    )

# Para cada valor, calcular % que o citou por grupo religioso
print("  Calculando valores prioritários por religião...")
va_results = []
for val_code, val_label in va01_labels.items():
    col = f'va_{val_code}'
    by_relig = df48_va.groupby('relig')[col].mean() * 100
    ev_pct = by_relig.get('Evangélico', 0)
    ca_pct = by_relig.get('Católico', 0)
    ou_pct = by_relig.get('Outro', 0)
    va_results.append({
        'codigo': val_code,
        'label': val_label,
        'evangelico': ev_pct,
        'catolico': ca_pct,
        'outro': ou_pct,
        'diff': ev_pct - ca_pct,
    })

va_results.sort(key=lambda x: abs(x['diff']), reverse=True)

print(f"  04748: {len(results_04748)} variáveis analisadas")

# ─── 3. NEW: 04752 — LGBTQIA+ (IPEC, Abr 2021, N=800) ───
print("\n[04752] LGBTQIA+ e Diversidade Sexual 2021...")
df52 = pd.read_csv(BASE / '04752.csv', encoding='utf-8')
relig52 = recode_religion_new(df52, 'RELIGIAO')

results_04752 = []

# P1: Tolerância de SP à população LGBTQIA+ (escala 1-10)
# Recategorizar: 1-3=intolerante, 4-6=neutro, 7-10=tolerante
df52_copy = df52.copy()
df52_copy['P1_cat'] = df52_copy['P1'].apply(lambda x:
    'Intolerante (1-3)' if pd.notna(x) and 1 <= int(float(x)) <= 3 else
    'Neutro (4-6)' if pd.notna(x) and 4 <= int(float(x)) <= 6 else
    'Tolerante (7-10)' if pd.notna(x) and 7 <= int(float(x)) <= 10 else
    np.nan
)
vl_p1 = {'Intolerante (1-3)': 'Intolerante (1-3)', 'Neutro (4-6)': 'Neutro (4-6)', 'Tolerante (7-10)': 'Tolerante (7-10)'}
r = crosstab_analysis(df52_copy, 'P1_cat', relig52, val_labels=vl_p1, skip_vals=[])
if r:
    r['tema'] = 'lgbtqia_genero'
    r['label'] = 'Tolerância da cidade de SP à população LGBTQIA+ (1-10, categorizado)'
    r['arquivo'] = '04752'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Abr/2021, São Paulo. Categorizado: 1-3=Intolerante, 4-6=Neutro, 7-10=Tolerante.'
    results_04752.append(r)

# P2: Administração municipal fez muito/pouco/nada para combater violência LGBTQIA+
vl_p2 = {1: 'Muito', 2: 'Pouco', 3: 'Nada'}
r = crosstab_analysis(df52, 'P2', relig52, val_labels=vl_p2, skip_vals=[98, 99])
if r:
    r['tema'] = 'lgbtqia_genero'
    r['label'] = 'Administração municipal fez muito/pouco/nada para combater violência LGBTQIA+'
    r['arquivo'] = '04752'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Abr/2021, São Paulo.'
    results_04752.append(r)

# P3: Importância de políticas LGBTQIA+
vl_p3 = {1: 'Muito importante', 2: 'Um pouco importante', 3: 'Pouco importante', 4: 'Nada importante'}
r = crosstab_analysis(df52, 'P3', relig52, val_labels=vl_p3, skip_vals=[98, 99])
if r:
    r['tema'] = 'lgbtqia_genero'
    r['label'] = 'Importância de políticas para promoção da igualdade LGBTQIA+'
    r['arquivo'] = '04752'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Abr/2021, São Paulo.'
    results_04752.append(r)

print(f"  04752: {len(results_04752)} variáveis analisadas")

# ─── 4. NEW: 04749 — Mulheres (IBOPE, Dez 2020, N=800) ───
print("\n[04749] Mulheres e Gênero 2020...")
df49 = pd.read_csv(BASE / '04749.csv', encoding='utf-8')
relig49 = recode_religion_new(df49, 'RELIGIAO')

results_04749 = []

# P04 A-F: Concordância com frases sobre gênero
vl_agree = {1: 'Concorda totalmente', 2: 'Concorda em parte',
             3: 'Discorda em parte', 4: 'Discorda totalmente'}
p04_labels = {
    'P04A': 'Os homens deveriam participar mais das tarefas domésticas',
    'P04B': 'A mulher tem o mesmo direito que o homem de trabalhar fora de casa',
    'P04C': 'É dever da mulher cuidar dos filhos e da casa',
    'P04D': 'Em caso de separação, os filhos devem ficar com a mãe',
    'P04E': 'O homem deve ser o principal provedor da família',
    'P04F': 'A mulher que trabalha fora cuida menos bem dos filhos',
}
for var, label in p04_labels.items():
    r = crosstab_analysis(df49, var, relig49, val_labels=vl_agree, skip_vals=[98, 99])
    if r:
        r['tema'] = 'lgbtqia_genero'
        r['label'] = label
        r['arquivo'] = '04749'
        r['ano'] = '2020'
        r['nota'] = 'IBOPE, Dez/2020, São Paulo. Concordância com afirmações sobre papéis de gênero.'
        results_04749.append(r)

# P06: Avaliação de assédio sexual/violência contra mulher
vl_p06 = {1: 'Aumentaram', 2: 'Se mantiveram', 3: 'Diminuíram'}
r = crosstab_analysis(df49, 'P06', relig49, val_labels=vl_p06, skip_vals=[98, 99])
if r:
    r['tema'] = 'lgbtqia_genero'
    r['label'] = 'Avaliação: assédio sexual e violência contra a mulher nos últimos 12 meses'
    r['arquivo'] = '04749'
    r['ano'] = '2020'
    r['nota'] = 'IBOPE, Dez/2020, São Paulo.'
    results_04749.append(r)

# P09: Sofreu discriminação no trabalho por ser mulher
vl_p09 = {1: 'Sim', 2: 'Não'}
r = crosstab_analysis(df49, 'P09', relig49, val_labels=vl_p09, skip_vals=[98, 99])
if r:
    r['tema'] = 'lgbtqia_genero'
    r['label'] = 'Já sofreu discriminação no trabalho por ser mulher'
    r['arquivo'] = '04749'
    r['ano'] = '2020'
    r['nota'] = 'IBOPE, Dez/2020, São Paulo.'
    results_04749.append(r)

print(f"  04749: {len(results_04749)} variáveis analisadas")

# ─── 5. NEW: 04746 — Meio Ambiente (IPEC, Abr 2021, N=800) ───
print("\n[04746] Meio Ambiente 2021...")
df46 = pd.read_csv(BASE / '04746.csv', encoding='utf-8')
relig46 = recode_religion_new(df46, 'RELIGIAO')
results_04746 = []

# P02: Importância do aquecimento global
vl_p02_46 = {1: 'Muito importante', 2: 'Pouco importante', 3: 'Nada importante'}
r = crosstab_analysis(df46, 'P02', relig46, val_labels=vl_p02_46, skip_vals=[98, 99])
if r:
    r['tema'] = 'cultura_politica'
    r['label'] = 'Importância do aquecimento global / mudanças climáticas'
    r['arquivo'] = '04746'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Abr/2021, São Paulo.'
    results_04746.append(r)

# P05: Causa dos problemas ambientais
vl_p05_46 = {1: 'Ação humana', 2: 'Mudanças naturais do meio ambiente', 3: 'Ambos', 97: 'Nenhum'}
r = crosstab_analysis(df46, 'P05', relig46, val_labels=vl_p05_46, skip_vals=[98, 99])
if r:
    r['tema'] = 'cultura_politica'
    r['label'] = 'Causa dos problemas ambientais: ação humana ou natureza?'
    r['arquivo'] = '04746'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Abr/2021, São Paulo. Relacionado a visão de mundo e ciência.'
    results_04746.append(r)

print(f"  04746: {len(results_04746)} variáveis analisadas")

# ─── 6. NEW: 04755 — Raça (IPEC, Jul 2021, N=800) ───
print("\n[04755] Raça e Racismo 2021...")
df55 = pd.read_csv(BASE / '04755.csv', encoding='utf-8')
relig55 = recode_religion_new(df55, 'RELIGIAO')
results_04755 = []

# P05 A-I: Concordância com frases sobre racismo
vl_agree_race = {1: 'Concorda totalmente', 2: 'Concorda em parte',
                  3: 'Discorda em parte', 4: 'Discorda totalmente'}
p05_labels = {
    'P5A': 'O racismo é um problema grave na cidade de SP',
    'P5B': 'Tenho medo de sofrer racismo na cidade de SP',
    'P5C': 'Já presenciei situações de racismo na minha vizinhança',
    'P5D': 'O racismo sistêmico (estrutural) existe no Brasil',
    'P5E': 'As cotas raciais nas universidades são necessárias',
    'P5F': 'As cotas raciais aumentam a divisão entre brancos e negros',
    'P5G': 'As pessoas negras precisam trabalhar mais para ser respeitadas',
    'P5H': 'O governo deve criar mais programas para reduzir a desigualdade racial',
    'P5I': 'As pessoas costumam exagerar nos casos de racismo',
}
for var, label in p05_labels.items():
    r = crosstab_analysis(df55, var, relig55, val_labels=vl_agree_race, skip_vals=[98, 99])
    if r:
        r['tema'] = 'raca_diversidade'
        r['label'] = label
        r['arquivo'] = '04755'
        r['ano'] = '2021'
        r['nota'] = 'IPEC, Jul/2021, São Paulo. Percepções sobre racismo e desigualdade racial.'
        results_04755.append(r)

print(f"  04755: {len(results_04755)} variáveis analisadas")

# ─── 7. NEW: 04767 — Mulheres (IPEC, Dez 2022, N=800) ───
print("\n[04767] Mulheres 2022...")
df67 = pd.read_csv(BASE / '04767.csv', encoding='utf-8')
relig67 = recode_religion_new(df67, 'RELIGIAO')
results_04767 = []

# P04: Onde mais corre risco de assédio
vl_p04_67 = {1: 'No transporte público', 2: 'Nos pontos de ônibus',
              3: 'Na rua', 4: 'Em bares e casas noturnas',
              5: 'No trabalho', 6: 'Em casa'}
r = crosstab_analysis(df67, 'P4', relig67, val_labels=vl_p04_67, skip_vals=[98, 99])
if r:
    r['tema'] = 'lgbtqia_genero'
    r['label'] = 'Local onde mais corre risco de sofrer assédio'
    r['arquivo'] = '04767'
    r['ano'] = '2022'
    r['nota'] = 'IPEC, Dez/2022, São Paulo.'
    results_04767.append(r)

# P05 A-F: Tipos de assédio sofridos
vl_sim_nao = {1: 'Sim', 2: 'Não'}
p05_67_labels = {
    'P5_1': 'Sofreu assédio no transporte coletivo',
    'P5_2': 'Sofreu assédio no transporte particular (taxi/Uber)',
    'P5_3': 'Sofreu assédio na rua',
    'P5_4': 'Sofreu assédio em bares ou casas noturnas',
    'P5_5': 'Sofreu assédio virtual (redes sociais)',
    'P5_6': 'Sofreu assédio no trabalho',
}
for var, label in p05_67_labels.items():
    r = crosstab_analysis(df67, var, relig67, val_labels=vl_sim_nao, skip_vals=[98, 99])
    if r:
        r['tema'] = 'lgbtqia_genero'
        r['label'] = label
        r['arquivo'] = '04767'
        r['ano'] = '2022'
        r['nota'] = 'IPEC, Dez/2022, São Paulo.'
        results_04767.append(r)

print(f"  04767: {len(results_04767)} variáveis analisadas")

# ─── 8. NEW: 04817 — Meio Ambiente (IPEC, Nov 2021, N=800) ───
print("\n[04817] Meio Ambiente 2021 (Nov)...")
df17 = pd.read_csv(BASE / '04817.csv', encoding='utf-8')
relig17 = recode_religion_new(df17, 'P_RELIGIAO')
results_04817 = []

vl_preoc = {1: 'Muito preocupado(a)', 2: 'Pouco preocupado(a)', 3: 'Nada preocupado(a)'}
r = crosstab_analysis(df17, 'P1', relig17, val_labels=vl_preoc, skip_vals=[98, 99])
if r:
    r['tema'] = 'cultura_politica'
    r['label'] = 'Preocupação com aquecimento global / mudanças climáticas'
    r['arquivo'] = '04817'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Nov/2021, São Paulo.'
    results_04817.append(r)

vl_causa = {1: 'Ação humana', 2: 'Mudanças naturais do meio ambiente', 3: 'Ambos', 97: 'Nenhum'}
r = crosstab_analysis(df17, 'P3', relig17, val_labels=vl_causa, skip_vals=[98, 99])
if r:
    r['tema'] = 'cultura_politica'
    r['label'] = 'Causa dos problemas ambientais: ação humana ou natureza? (Nov/2021)'
    r['arquivo'] = '04817'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Nov/2021, São Paulo.'
    results_04817.append(r)

print(f"  04817: {len(results_04817)} variáveis analisadas")

# ─── 9. NEW: 04770 — Violência Doméstica (IPEC, Dez 2021, N=800) ───
print("\n[04770] Violência Doméstica 2021...")
df70 = pd.read_csv(BASE / '04770.csv', encoding='utf-8')
relig70 = recode_religion_new(df70, 'RELIGIAO')
results_04770 = []

vl_violencia = {1: 'Aumentou muito', 2: 'Aumentou um pouco', 3: 'Está igual',
                4: 'Diminuiu um pouco', 5: 'Diminuiu muito'}
r = crosstab_analysis(df70, 'P3', relig70, val_labels=vl_violencia, skip_vals=[98, 99])
if r:
    r['tema'] = 'lgbtqia_genero'
    r['label'] = 'Percepção: violência doméstica contra mulheres nos últimos 12 meses'
    r['arquivo'] = '04770'
    r['ano'] = '2021'
    r['nota'] = 'IPEC, Dez/2021, São Paulo.'
    results_04770.append(r)

print(f"  04770: {len(results_04770)} variáveis analisadas")


# ─────────────────────────────────────────────
# GENERATE MARKDOWN OUTPUT
# ─────────────────────────────────────────────
print("\n\nGerando documento Markdown...")

all_new_results = (results_04748 + results_04752 + results_04749 +
                   results_04746 + results_04755 + results_04767 +
                   results_04817 + results_04770)

# Organize by theme
themed = {}
for r in all_new_results:
    t = r.get('tema', 'cultura_politica')
    themed.setdefault(t, []).append(r)

# Theme labels
theme_labels = {
    'eleicoes_democracia': '## 1. Eleições e Democracia',
    'cultura_politica': '## 2. Cultura Política e Valores Cívicos',
    'confianca_institucional': '## 3. Confiança Institucional',
    'valores_morais': '## 4. Valores Morais e Família',
    'lgbtqia_genero': '## 5. Gênero, Sexualidade e LGBTQIA+',
    'raca_diversidade': '## 6. Raça, Diversidade e Inclusão',
    'pratica_religiosa': '## 7. Prática Religiosa e Identidade',
    'identidade_publica': '## 8. Identidade Religiosa no Espaço Público',
}

lines = []
lines.append("# Ilustrações Quantitativas CESOP para a Tese")
lines.append("## Opinião Pública por Religião no Brasil (1993–2023)")
lines.append("")
lines.append("**Nota metodológica:** Todas as tabelas apresentam percentuais calculados dentro de cada grupo religioso (% coluna), permitindo comparação direta entre Evangélicos, Católicos e Outros. O teste qui-quadrado (χ²) é aplicado apenas na comparação Evangélicos × Católicos. Significância: *** p<0,001 | ** p<0,01 | * p<0,05 | n.s. = não significativo.")
lines.append("")
lines.append("**Classificação religiosa (pesquisas RNSP/IPEC 2020–2022):** 1=Católico; 2–11=Evangélico (todas as denominações: Assembleia de Deus, Batista, Universal, Deus é Amor, Quadrangular, IIGD, Renascer, Sara, outras); 12=Adventista; 19=Sem religião/Agnóstico; 20=Ateu.")
lines.append("")
lines.append("---")
lines.append("")

# ────── Section A: Existing report summary (top findings) ──────
lines.append("## Síntese das Análises Históricas (1993–2020)")
lines.append("")
lines.append("*Dados das 77 variáveis já analisadas nas pesquisas CESOP (1993–2020), disponíveis no arquivo `cesop_relatorio_religiao.txt`. Abaixo apresentam-se os achados mais relevantes organizados por tema.*")
lines.append("")

if existing:
    # Group by theme using keyword matching
    old_themed = {}
    for r in existing:
        if 'tema' not in r:
            r['tema'] = classify_theme(r.get('tema', '') + ' ' + r.get('dado_principal', ''))
        tema = classify_theme(r.get('dado_principal', '') + ' ' + r.get('tema', ''))
        old_themed.setdefault(tema, []).append(r)

    # Summarize key findings from existing report
    lines.append("### Valores Morais e Família (pesquisas 1993–2020)")
    lines.append("")

    moral_results = [r for r in existing if any(k in r.get('tema', '').lower() for k in
                     ['virgem', 'aborto', 'família', 'famil', 'casamento', 'sexualid', 'moral'])]

    for r in moral_results[:12]:
        sig = r.get('sig', '')
        p_val = r.get('p_value', None)
        chi_str = r.get('chi2_str', '')
        arquivo = r.get('arquivo', '?')
        tema = r.get('tema', '?')
        n_t = r.get('n_total', '?')
        n_e = r.get('n_evang', '?')
        n_c = r.get('n_cat', '?')
        dado = r.get('dado_principal', '')

        lines.append(f"**{tema}**")
        lines.append(f"*{arquivo} | N={n_t} (Ev={n_e}, Cat={n_c})*")
        if chi_str:
            lines.append(f"*Qui-quadrado: {chi_str}*")
        if dado:
            lines.append(f"> {dado}")
        lines.append("")

    lines.append("### Confiança Institucional e Igrejas (pesquisas 1993–2020)")
    lines.append("")
    conf_results = [r for r in existing if any(k in r.get('tema', '').lower() for k in
                    ['confiança', 'avaliação', 'ics', 'cara da'])]
    for r in conf_results[:10]:
        tema = r.get('tema', '?')
        arquivo = r.get('arquivo', '?')
        n_t = r.get('n_total', '?')
        n_e = r.get('n_evang', '?')
        n_c = r.get('n_cat', '?')
        dado = r.get('dado_principal', '')
        chi_str = r.get('chi2_str', '')
        lines.append(f"**{tema}**")
        lines.append(f"*{arquivo} | N={n_t} (Ev={n_e}, Cat={n_c})*")
        if chi_str:
            lines.append(f"*Qui-quadrado: {chi_str}*")
        if dado:
            lines.append(f"> {dado}")
        lines.append("")

    lines.append("### Cultura Política e Democracia (pesquisas 1993–2020)")
    lines.append("")
    pol_results = [r for r in existing if any(k in r.get('tema', '').lower() for k in
                   ['democracia', 'ditadura', 'partidos', 'governo', 'político', 'esquerda', 'corrupção', 'orgulho', 'ideológico'])]
    for r in pol_results[:10]:
        tema = r.get('tema', '?')
        arquivo = r.get('arquivo', '?')
        n_t = r.get('n_total', '?')
        n_e = r.get('n_evang', '?')
        n_c = r.get('n_cat', '?')
        dado = r.get('dado_principal', '')
        chi_str = r.get('chi2_str', '')
        lines.append(f"**{tema}**")
        lines.append(f"*{arquivo} | N={n_t} (Ev={n_e}, Cat={n_c})*")
        if chi_str:
            lines.append(f"*Qui-quadrado: {chi_str}*")
        if dado:
            lines.append(f"> {dado}")
        lines.append("")

    lines.append("### Identidade e Prática Religiosa (pesquisas 1993–2020)")
    lines.append("")
    id_results = [r for r in existing if any(k in r.get('tema', '').lower() for k in
                  ['identidade', 'declarar', 'à vontade', 'frequência', 'missa', 'culto', 'grupo religioso', 'membro', 'religiosidade', 'importância da religião', 'preconceito por causa'])]
    for r in id_results[:12]:
        tema = r.get('tema', '?')
        arquivo = r.get('arquivo', '?')
        n_t = r.get('n_total', '?')
        n_e = r.get('n_evang', '?')
        n_c = r.get('n_cat', '?')
        dado = r.get('dado_principal', '')
        chi_str = r.get('chi2_str', '')
        lines.append(f"**{tema}**")
        lines.append(f"*{arquivo} | N={n_t} (Ev={n_e}, Cat={n_c})*")
        if chi_str:
            lines.append(f"*Qui-quadrado: {chi_str}*")
        if dado:
            lines.append(f"> {dado}")
        lines.append("")

lines.append("---")
lines.append("")

# ────── Section B: New analyses (RNSP/IPEC 2020-2022) ──────

# Theme order for new results
theme_order = [
    'eleicoes_democracia',
    'cultura_politica',
    'confianca_institucional',
    'valores_morais',
    'lgbtqia_genero',
    'raca_diversidade',
    'pratica_religiosa',
    'identidade_publica',
]

lines.append("## Análises Novas — Pesquisas RNSP/IPEC (2020–2022)")
lines.append("")
lines.append("*Análises realizadas para esta tese com as pesquisas mais recentes do acervo CESOP/RNSP.*")
lines.append("")

for theme in theme_order:
    if theme not in themed:
        continue
    theme_results = themed[theme]
    if not theme_results:
        continue

    lines.append(theme_labels.get(theme, f"## {theme}"))
    lines.append("")

    for r in theme_results:
        md = format_result_md(
            r,
            title=r.get('label', r.get('var', '?')),
            arquivo=r.get('arquivo', '?'),
            ano=r.get('ano', '?'),
            note=r.get('nota', '')
        )
        if md:
            lines.append(md)
            lines.append("")

# ────── Section C: Social values ranking ──────
lines.append("---")
lines.append("")
lines.append("## Apêndice: Valores Sociais Prioritários por Religião (2021)")
lines.append("")
lines.append("*Pesquisa IPEC/CESOP 04748, Set/2021, N=1.009 entrevistas nacionais.*")
lines.append("*Pergunta: \"Qual desses valores é mais importante para você?\" (citação entre as 3 primeiras escolhas)*")
lines.append("")
lines.append("| Valor | Evangélicos | Católicos | Outros | Δ (Ev–Cat) |")
lines.append("|-------|-------------|-----------|--------|------------|")

for va in va_results:
    diff_str = f"+{va['diff']:.1f}pp" if va['diff'] > 0 else f"{va['diff']:.1f}pp"
    lines.append(f"| {va['label']} | {va['evangelico']:.1f}% | {va['catolico']:.1f}% | {va['outro']:.1f}% | {diff_str} |")

lines.append("")
lines.append("> **Nota:** Percentual de respondentes que citou o valor em alguma das 3 posições de prioridade.")
lines.append("")

# ────── Summary stats ──────
lines.append("---")
lines.append("")
lines.append("## Resumo Estatístico das Novas Análises")
lines.append("")

total_new = len(all_new_results)
sig_new = [r for r in all_new_results if r.get('p_value') is not None and r['p_value'] < 0.05]
strong_sig = [r for r in all_new_results if r.get('p_value') is not None and r['p_value'] < 0.001]

lines.append(f"- **Total de variáveis analisadas (novas):** {total_new}")
lines.append(f"- **Com diferença estatisticamente significativa (p<0,05):** {len(sig_new)} ({100*len(sig_new)/max(total_new,1):.0f}%)")
lines.append(f"- **Com diferença muito significativa (p<0,001):** {len(strong_sig)} ({100*len(strong_sig)/max(total_new,1):.0f}%)")
lines.append(f"- **Análises do relatório histórico (1993–2020):** {len(existing)}")
lines.append(f"- **Total geral de análises:** {total_new + len(existing)}")
lines.append("")

# Save
output_path = OUT / 'cesop_ilustracoes_tese.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\n✓ Documento salvo: {output_path}")
print(f"  Novas análises: {total_new} variáveis")
print(f"  Significativas (p<0.05): {len(sig_new)}")
print(f"  Muito significativas (p<0.001): {len(strong_sig)}")
print(f"  Históricas (relatório existente): {len(existing)}")

# Also print key findings to console
print("\n" + "="*70)
print("ACHADOS MAIS RELEVANTES PARA A TESE")
print("="*70)

# Top findings by significance
sig_sorted = sorted([r for r in all_new_results if r.get('p_value') is not None],
                    key=lambda x: x['p_value'])

for r in sig_sorted[:20]:
    ct = r.get('ct_pct')
    if ct is None or 'Evangélico' not in ct.columns or 'Católico' not in ct.columns:
        continue

    vl = r.get('val_labels', {})
    diffs = {}
    for val in ct.index:
        diffs[val] = abs(ct.loc[val, 'Evangélico'] - ct.loc[val, 'Católico'])
    if not diffs:
        continue
    mv = max(diffs, key=diffs.get)
    ml = vl.get(mv, vl.get(float(mv) if not isinstance(mv, str) else mv, str(mv)))
    ev = ct.loc[mv, 'Evangélico']
    ca = ct.loc[mv, 'Católico']

    print(f"\n[{r['arquivo']}/{r['ano']}] {r['label'][:70]}...")
    print(f"  p={r['p_value']:.4f} {sig_stars(r['p_value'])} | '{ml}': Ev={ev:.1f}% vs Cat={ca:.1f}% (Δ={abs(ev-ca):.1f}pp)")
