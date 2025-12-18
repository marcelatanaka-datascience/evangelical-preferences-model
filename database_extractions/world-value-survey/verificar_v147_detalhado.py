"""
Script para verificar detalhadamente a variável V147
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

file_path = BASE_DIR / 'wave-2-1991-br' / 'WV2_Data_Brazil_Csv_v1.6.1.csv'

print("="*70)
print("VERIFICAÇÃO DETALHADA DA VARIÁVEL V147")
print("="*70)

df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)

print(f"\nArquivo: {file_path}")
print(f"Total de linhas: {len(df)}")
print(f"Total de colunas: {len(df.columns)}")

# Verificar se V147 existe
if 'V147' not in df.columns:
    print("\n❌ V147 não encontrada nas colunas!")
    print(f"Colunas próximas: {[c for c in df.columns if '147' in str(c) or '146' in str(c) or '148' in str(c)]}")
else:
    print(f"\n✓ V147 encontrada no índice: {list(df.columns).index('V147')}")
    
    # Valores únicos
    unique_vals = sorted([x for x in df['V147'].unique() if pd.notna(x)])
    print(f"\nValores únicos de V147: {unique_vals}")
    
    # Frequência completa
    print("\nFrequência completa:")
    freq = df['V147'].value_counts().sort_index()
    for val, count in freq.items():
        pct = (count / len(df)) * 100
        print(f"  {val}: {count} ({pct:.2f}%)")
    
    # Verificar se há valores > 2
    maiores_que_2 = (df['V147'] > 2).sum()
    print(f"\nValores > 2: {maiores_que_2}")
    
    # Verificar se há valores entre 1 e 8
    entre_1_e_8 = ((df['V147'] >= 1) & (df['V147'] <= 8)).sum()
    print(f"Valores entre 1 e 8: {entre_1_e_8}")
    
    # Mostrar primeiras 50 linhas
    print("\nPrimeiras 50 linhas da V147:")
    print(df['V147'].head(50).tolist())
    
    # Verificar colunas próximas
    idx = list(df.columns).index('V147')
    print(f"\nColunas próximas à V147 (índices {idx-2} a {idx+2}):")
    for i in range(max(0, idx-2), min(len(df.columns), idx+3)):
        col = df.columns[i]
        if col != 'V147':
            vals = sorted([x for x in df[col].unique() if pd.notna(x)])[:10]
            print(f"  {i}: {col} - primeiros valores únicos: {vals}")

print("\n" + "="*70)







