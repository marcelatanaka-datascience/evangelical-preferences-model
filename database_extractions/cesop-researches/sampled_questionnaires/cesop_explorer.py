#!/usr/bin/env python3
"""
CESOP Explorer — Mapeia variáveis de todas as pesquisas selecionadas
=====================================================================
Este script abre cada arquivo .sav das pesquisas do CESOP e gera um
relatório com todas as variáveis, seus labels e distribuições.

USO:
1. Coloque todos os .sav em uma pasta (ex: /cesop_data/)
2. Rode: python cesop_explorer.py /cesop_data/
3. O script gera um CSV com o mapeamento de todas as variáveis

O relatório permite identificar rapidamente:
- Qual variável contém a religião do entrevistado
- Quais perguntas são relevantes para a tese (moralidade, confiança, etc.)
"""

import pyreadstat
import pandas as pd
import os
import sys
from pathlib import Path


def explorar_pesquisa(sav_path):
    """
    Explora um arquivo .sav e retorna DataFrame com todas as variáveis,
    seus labels, tipos e distribuições.
    """
    try:
        df, meta = pyreadstat.read_sav(str(sav_path))
    except Exception as e:
        print(f"  ERRO ao carregar {sav_path.name}: {e}")
        return None

    resultados = []

    for col in df.columns:
        # Info básica
        n_valid = df[col].notna().sum()
        n_unique = df[col].nunique()
        dtype = str(df[col].dtype)

        # Label da variável (se existir nos metadados)
        var_label = ""
        if hasattr(meta, 'column_names_to_labels'):
            var_label = meta.column_names_to_labels.get(col, "") or ""

        # Labels dos valores (se existirem)
        val_labels = {}
        if hasattr(meta, 'variable_value_labels'):
            val_labels = meta.variable_value_labels.get(col, {})

        # Distribuição (top 10 valores)
        vc = df[col].value_counts().sort_index().head(10)
        dist_str = "; ".join([f"{v}={c}" for v, c in vc.items()])

        # Labels dos valores formatados
        val_labels_str = "; ".join([f"{v}={l}" for v, l in val_labels.items()]) if val_labels else ""

        # Flag: possível variável de religião?
        is_religion = False
        religion_keywords = ['relig', 'igreja', 'culto', 'denom', 'credo', 'fe ', 'fé ',
                            'evangel', 'catol', 'protest', 'pentecost', 'crença']
        text_to_check = (var_label + " " + val_labels_str + " " + col).lower()
        for kw in religion_keywords:
            if kw in text_to_check:
                is_religion = True
                break

        # Flag: possível variável de interesse temático?
        is_theme = False
        theme_keywords = ['abort', 'homos', 'gay', 'lésbic', 'casamento', 'família',
                         'moral', 'sexual', 'gênero', 'gender', 'confiança', 'confia',
                         'democra', 'congresso', 'justiça', 'governo', 'partido',
                         'igreja', 'imprensa', 'mulher', 'divers', 'preconceito',
                         'racis', 'pena de morte', 'droga', 'prostit', 'pornograf',
                         'ensino religioso', 'liberdade', 'perseguição']
        for kw in theme_keywords:
            if kw in text_to_check:
                is_theme = True
                break

        resultados.append({
            'arquivo': sav_path.name,
            'variavel': col,
            'label_variavel': var_label,
            'tipo': dtype,
            'n_valid': n_valid,
            'n_unique': n_unique,
            'distribuicao': dist_str,
            'labels_valores': val_labels_str,
            'flag_religiao': is_religion,
            'flag_tema': is_theme
        })

    return pd.DataFrame(resultados)


def main():
    if len(sys.argv) < 2:
        print("Uso: python cesop_explorer.py <pasta_com_sav>")
        print("Exemplo: python cesop_explorer.py /cesop_data/")
        sys.exit(1)

    pasta = Path(sys.argv[1])
    if not pasta.exists():
        print(f"Pasta não encontrada: {pasta}")
        sys.exit(1)

    # Encontrar todos os .sav (incluindo .SAV, .saav e outras variantes)
    arquivos = []
    for pattern in ["*.sav", "*.SAV", "*.Sav", "*.saav", "*.SAAV"]:
        arquivos.extend(pasta.glob(pattern))
    if not arquivos:
        for pattern in ["*.sav", "*.SAV", "*.Sav", "*.saav", "*.SAAV"]:
            arquivos.extend(pasta.rglob(pattern))
    # Remover duplicatas e ordenar
    arquivos = sorted(set(arquivos), key=lambda x: x.name)

    if not arquivos:
        print(f"Nenhum arquivo .sav encontrado em {pasta}")
        sys.exit(1)

    print(f"Encontrados {len(arquivos)} arquivos .sav")
    print("=" * 70)

    todos_resultados = []

    for i, arq in enumerate(arquivos, 1):
        print(f"\n[{i}/{len(arquivos)}] Processando: {arq.name}")
        resultado = explorar_pesquisa(arq)
        if resultado is not None:
            todos_resultados.append(resultado)
            n_vars = len(resultado)
            n_relig = resultado['flag_religiao'].sum()
            n_tema = resultado['flag_tema'].sum()
            print(f"  {n_vars} variáveis | {n_relig} possíveis religião | {n_tema} possíveis tema")

    if not todos_resultados:
        print("\nNenhuma pesquisa processada com sucesso.")
        sys.exit(1)

    # Consolidar
    df_all = pd.concat(todos_resultados, ignore_index=True)

    # Salvar relatório completo
    output_full = pasta / "cesop_mapeamento_completo.csv"
    df_all.to_csv(output_full, index=False, encoding='utf-8-sig')
    print(f"\nRelatório completo salvo: {output_full}")
    print(f"  Total: {len(df_all)} variáveis em {len(arquivos)} pesquisas")

    # Salvar relatório filtrado (apenas variáveis de religião e tema)
    df_filtered = df_all[df_all['flag_religiao'] | df_all['flag_tema']].copy()
    output_filtered = pasta / "cesop_variaveis_relevantes.csv"
    df_filtered.to_csv(output_filtered, index=False, encoding='utf-8-sig')
    print(f"\nVariáveis relevantes salvas: {output_filtered}")
    print(f"  Total: {len(df_filtered)} variáveis filtradas")

    # Resumo por pesquisa
    print("\n" + "=" * 70)
    print("RESUMO POR PESQUISA")
    print("=" * 70)
    for arq_name in df_all['arquivo'].unique():
        subset = df_all[df_all['arquivo'] == arq_name]
        relig = subset[subset['flag_religiao']]
        tema = subset[subset['flag_tema']]
        print(f"\n{arq_name}:")
        if len(relig) > 0:
            print(f"  RELIGIÃO:")
            for _, row in relig.iterrows():
                print(f"    {row['variavel']}: {row['label_variavel']}")
                if row['labels_valores']:
                    print(f"      Valores: {row['labels_valores'][:120]}")
        if len(tema) > 0:
            print(f"  TEMAS:")
            for _, row in tema.iterrows():
                print(f"    {row['variavel']}: {row['label_variavel'][:80]}")


if __name__ == "__main__":
    main()