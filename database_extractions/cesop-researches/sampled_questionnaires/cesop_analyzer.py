#!/usr/bin/env python3
"""
CESOP Analyzer — Extrai dados por religião de todas as pesquisas mapeadas
=========================================================================
Este script usa o mapeamento gerado pelo cesop_explorer.py para extrair
tabelas cruzadas (religião x variável temática) de todas as pesquisas.

USO:
1. Rode cesop_explorer.py primeiro para gerar o mapeamento
2. Preencha o arquivo cesop_config.csv com o mapeamento correto
3. Rode: python cesop_analyzer.py

O config.csv deve ter o formato:
arquivo,var_religiao,cod_evangelico,cod_catolico,var_tema,label_tema

Exemplo:
00828.sav,p24,"4,5","2,3",p18,Posição sobre aborto
00853.sav,religiao,"4","2,3",p19,Valores familiares
"""

import pyreadstat
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')


def recodificar_religiao(df, var_religiao, cod_evangelico, cod_catolico):
    """
    Recodifica variável de religião em categorias padronizadas.
    
    Parameters:
        df: DataFrame
        var_religiao: nome da coluna de religião
        cod_evangelico: string com códigos separados por vírgula (ex: "4,5")
        cod_catolico: string com códigos separados por vírgula (ex: "2,3")
    
    Returns:
        Series com categorias: Evangélico, Católico, Outro, NR/NS
    """
    # Parsear códigos
    evang_codes = [float(x.strip()) for x in str(cod_evangelico).split(',')]
    cat_codes = [float(x.strip()) for x in str(cod_catolico).split(',')]
    
    def classificar(val):
        if pd.isna(val):
            return 'NR/NS'
        if val in evang_codes:
            return 'Evangélico'
        if val in cat_codes:
            return 'Católico'
        return 'Outro'
    
    return df[var_religiao].apply(classificar)


def analisar_pesquisa(sav_path, var_religiao, cod_evangelico, cod_catolico, 
                      var_tema, label_tema):
    """
    Analisa uma pesquisa: crosstab religião x tema + qui-quadrado.
    
    Returns:
        dict com resultados ou None em caso de erro
    """
    try:
        df, meta = pyreadstat.read_sav(str(sav_path))
    except Exception as e:
        return {'erro': f"Erro ao carregar: {e}"}
    
    # Verificar se variáveis existem
    if var_religiao not in df.columns:
        return {'erro': f"Variável de religião '{var_religiao}' não encontrada"}
    if var_tema not in df.columns:
        return {'erro': f"Variável temática '{var_tema}' não encontrada"}
    
    # Recodificar religião
    df['religiao_recodificada'] = recodificar_religiao(
        df, var_religiao, cod_evangelico, cod_catolico
    )
    
    # Labels da variável temática
    val_labels = {}
    if hasattr(meta, 'variable_value_labels'):
        val_labels = meta.variable_value_labels.get(var_tema, {})
    
    # Label da variável temática
    var_tema_label = ""
    if hasattr(meta, 'column_names_to_labels'):
        var_tema_label = meta.column_names_to_labels.get(var_tema, "")
    
    # Crosstab: contagens
    ct_counts = pd.crosstab(df[var_tema], df['religiao_recodificada'], margins=True)
    
    # Crosstab: percentuais por coluna (dentro de cada religião)
    ct_pct = pd.crosstab(df[var_tema], df['religiao_recodificada'], normalize='columns') * 100
    
    # Qui-quadrado (excluindo NR/NS e margem)
    ct_for_chi = pd.crosstab(
        df[df['religiao_recodificada'].isin(['Evangélico', 'Católico'])][var_tema],
        df[df['religiao_recodificada'].isin(['Evangélico', 'Católico'])]['religiao_recodificada']
    )
    
    chi2, p_value, dof, expected = None, None, None, None
    if ct_for_chi.shape[0] > 1 and ct_for_chi.shape[1] > 1:
        try:
            chi2, p_value, dof, expected = chi2_contingency(ct_for_chi)
        except Exception:
            pass
    
    # Contagem por grupo
    n_evang = (df['religiao_recodificada'] == 'Evangélico').sum()
    n_cat = (df['religiao_recodificada'] == 'Católico').sum()
    n_outro = (df['religiao_recodificada'] == 'Outro').sum()
    n_total = len(df)
    
    return {
        'arquivo': sav_path.name,
        'label_tema': label_tema,
        'var_tema': var_tema,
        'var_tema_label': var_tema_label,
        'val_labels': val_labels,
        'n_total': n_total,
        'n_evangelico': n_evang,
        'n_catolico': n_cat,
        'n_outro': n_outro,
        'crosstab_counts': ct_counts,
        'crosstab_pct': ct_pct,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
    }


def gerar_relatorio(resultados, output_path):
    """Gera relatório consolidado em texto."""
    lines = []
    lines.append("=" * 80)
    lines.append("RELATÓRIO CESOP — OPINIÃO PÚBLICA POR RELIGIÃO")
    lines.append("Gerado para incorporação ao argumento histórico da tese")
    lines.append("=" * 80)
    
    for r in resultados:
        lines.append("")
        lines.append("-" * 80)
        
        if 'erro' in r:
            lines.append(f"ARQUIVO: {r.get('arquivo', '???')}")
            lines.append(f"  ERRO: {r['erro']}")
            continue
        
        lines.append(f"ARQUIVO: {r['arquivo']}")
        lines.append(f"TEMA: {r['label_tema']}")
        lines.append(f"VARIÁVEL: {r['var_tema']} — {r['var_tema_label']}")
        lines.append(f"N total={r['n_total']} | Evangélicos={r['n_evangelico']} | "
                     f"Católicos={r['n_catolico']} | Outros={r['n_outro']}")
        
        # Qui-quadrado
        if r['chi2'] is not None:
            sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else \
                  "*" if r['p_value'] < 0.05 else "n.s."
            lines.append(f"Qui-quadrado: χ²={r['chi2']:.2f}, p={r['p_value']:.4f} ({sig}), gl={r['dof']}")
        
        # Tabela de percentuais
        lines.append("")
        lines.append("  PERCENTUAIS POR GRUPO RELIGIOSO:")
        
        ct = r['crosstab_pct']
        # Adicionar labels dos valores se disponíveis
        for val in ct.index:
            val_label = r['val_labels'].get(val, str(val))
            row_data = []
            for col in ['Evangélico', 'Católico', 'Outro']:
                if col in ct.columns:
                    row_data.append(f"{col}={ct.loc[val, col]:.1f}%")
            lines.append(f"    {val_label}: {' | '.join(row_data)}")
        
        # Dado principal para incorporação no texto
        lines.append("")
        lines.append("  DADO PRINCIPAL PARA O TEXTO:")
        if 'Evangélico' in ct.columns and 'Católico' in ct.columns:
            # Identificar a maior diferença entre evangélicos e católicos
            diffs = {}
            for val in ct.index:
                diff = abs(ct.loc[val, 'Evangélico'] - ct.loc[val, 'Católico'])
                diffs[val] = diff
            if diffs:
                max_diff_val = max(diffs, key=diffs.get)
                val_label = r['val_labels'].get(max_diff_val, str(max_diff_val))
                ev_pct = ct.loc[max_diff_val, 'Evangélico']
                cat_pct = ct.loc[max_diff_val, 'Católico']
                lines.append(f"    Maior diferença: '{val_label}' → "
                           f"Evangélicos={ev_pct:.1f}% vs Católicos={cat_pct:.1f}% "
                           f"(diferença={abs(ev_pct-cat_pct):.1f}pp)")
    
    # Salvar
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nRelatório salvo: {output_path}")


def main():
    """
    Função principal.
    
    Espera um arquivo cesop_config.csv na mesma pasta dos .sav com formato:
    arquivo,var_religiao,cod_evangelico,cod_catolico,var_tema,label_tema
    """
    if len(sys.argv) < 2:
        print("Uso: python cesop_analyzer.py <pasta_com_sav_e_config>")
        print("\nO arquivo cesop_config.csv deve estar na pasta e ter o formato:")
        print("arquivo,var_religiao,cod_evangelico,cod_catolico,var_tema,label_tema")
        print("\nExemplo de linha:")
        print('00828.sav,p24,"4,5","2,3",p18,Posição sobre aborto')
        sys.exit(1)
    
    pasta = Path(sys.argv[1])
    config_path = pasta / "cesop_config.csv"
    
    if not config_path.exists():
        print(f"Arquivo de configuração não encontrado: {config_path}")
        print("Rode cesop_explorer.py primeiro e depois crie o cesop_config.csv")
        sys.exit(1)
    
    config = pd.read_csv(config_path, encoding='utf-8-sig')
    print(f"Config carregado: {len(config)} pesquisas para analisar")
    print("=" * 70)
    
    resultados = []
    
    for i, row in config.iterrows():
        sav_path = pasta / row['arquivo']
        print(f"\n[{i+1}/{len(config)}] {row['arquivo']} — {row['label_tema']}")
        
        if not sav_path.exists():
            print(f"  AVISO: arquivo não encontrado")
            resultados.append({'arquivo': row['arquivo'], 'erro': 'Arquivo não encontrado'})
            continue
        
        r = analisar_pesquisa(
            sav_path,
            var_religiao=row['var_religiao'],
            cod_evangelico=str(row['cod_evangelico']),
            cod_catolico=str(row['cod_catolico']),
            var_tema=row['var_tema'],
            label_tema=row['label_tema']
        )
        r['arquivo'] = row['arquivo']
        resultados.append(r)
        
        # Print resumo
        if 'erro' not in r:
            if r['chi2'] is not None:
                sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else \
                      "*" if r['p_value'] < 0.05 else "n.s."
                print(f"  N={r['n_total']} | Ev={r['n_evangelico']} | Cat={r['n_catolico']} | "
                      f"χ²={r['chi2']:.1f} ({sig})")
            else:
                print(f"  N={r['n_total']} | Ev={r['n_evangelico']} | Cat={r['n_catolico']}")
        else:
            print(f"  {r['erro']}")
    
    # Gerar relatório
    output_path = pasta / "cesop_relatorio_religiao.txt"
    gerar_relatorio(resultados, output_path)
    
    # Também salvar dados resumidos em CSV
    resumo = []
    for r in resultados:
        if 'erro' in r:
            resumo.append({
                'arquivo': r.get('arquivo', ''),
                'erro': r['erro']
            })
        else:
            resumo.append({
                'arquivo': r['arquivo'],
                'tema': r['label_tema'],
                'n_total': r['n_total'],
                'n_evangelico': r['n_evangelico'],
                'n_catolico': r['n_catolico'],
                'chi2': r['chi2'],
                'p_value': r['p_value'],
            })
    
    df_resumo = pd.DataFrame(resumo)
    resumo_path = pasta / "cesop_resumo.csv"
    df_resumo.to_csv(resumo_path, index=False, encoding='utf-8-sig')
    print(f"Resumo CSV salvo: {resumo_path}")


if __name__ == "__main__":
    main()
