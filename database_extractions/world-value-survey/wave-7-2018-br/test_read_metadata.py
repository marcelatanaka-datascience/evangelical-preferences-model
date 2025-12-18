#!/usr/bin/env python3
"""Teste para ler apenas metadados do arquivo WVS"""

import pyreadstat

sav_file_path = "WVS_Wave_7_Brazil_Spss_v5.1.sav"

try:
    print("Tentando ler apenas metadados...")
    df_empty, meta = pyreadstat.read_sav(sav_file_path, metadataonly=True)
    print(f"Sucesso! Variáveis encontradas: {len(df_empty.columns)}")
    print("\nPrimeiras 50 variáveis:")
    for i, col in enumerate(df_empty.columns[:50], 1):
        print(f"  {i}. {col}")
    
    # Procurar variáveis de religião
    religiao_vars = [col for col in df_empty.columns if 'relig' in col.lower() or 'Q289' in col.upper()]
    print(f"\nVariáveis de religião: {religiao_vars}")
    
    # Procurar Q20 e Q21
    q20_vars = [col for col in df_empty.columns if col.upper().startswith('Q20')]
    q21_vars = [col for col in df_empty.columns if col.upper().startswith('Q21')]
    print(f"Variáveis Q20: {q20_vars}")
    print(f"Variáveis Q21: {q21_vars}")
    
except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()










