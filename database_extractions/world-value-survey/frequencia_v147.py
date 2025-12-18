"""
Script para gerar frequências simples da variável V147
nos arquivos original e com_freq_inv da Wave 2.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Arquivos
file_original = BASE_DIR / 'wave-2-1991-br' / 'WV2_Data_Brazil_Csv_v1.6.1.csv'
file_freq_inv = BASE_DIR / 'wave-2-1991-br' / 'WV2_Data_Brazil_Csv_v1.6.1_com_freq_inv.csv'

print("="*70)
print("FREQUÊNCIAS DA VARIÁVEL V147 - WAVE 2 (1991)")
print("="*70)

# Carregar arquivo original
print("\n[ARQUIVO ORIGINAL]")
print("-"*70)
try:
    # Ler usando csv.reader para lidar com desalinhamento de colunas
    import csv
    rows = []
    with open(file_original, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        for row in reader:
            if len(row) > len(header):
                row = row[:len(header)]
            elif len(row) < len(header):
                row.extend([''] * (len(header) - len(row)))
            rows.append(row)
    df_original = pd.DataFrame(rows, columns=header)
    for col in df_original.columns:
        try:
            df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
        except:
            pass
    
    if 'V147' in df_original.columns:
        freq_original = df_original['V147'].value_counts().sort_index()
        freq_original_pct = df_original['V147'].value_counts(normalize=True).sort_index() * 100
        
        print(f"\nTotal de casos: {len(df_original)}")
        print(f"Valores missing: {df_original['V147'].isna().sum()}")
        print(f"Valores não-missing: {df_original['V147'].notna().sum()}")
        
        print("\nFrequência absoluta:")
        print(freq_original.to_string())
        
        print("\nFrequência percentual:")
        for val, count in freq_original.items():
            pct = freq_original_pct[val]
            print(f"  {val}: {pct:.2f}%")
    else:
        print("Variável V147 não encontrada no arquivo original")
        
except Exception as e:
    print(f"Erro ao carregar arquivo original: {e}")

# Carregar arquivo com_freq_inv
print("\n\n[ARQUIVO COM_FREQ_INV]")
print("-"*70)
try:
    # Ler usando csv.reader para lidar com desalinhamento de colunas
    import csv
    rows = []
    with open(file_freq_inv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        for row in reader:
            if len(row) > len(header):
                row = row[:len(header)]
            elif len(row) < len(header):
                row.extend([''] * (len(header) - len(row)))
            rows.append(row)
    df_freq_inv = pd.DataFrame(rows, columns=header)
    for col in df_freq_inv.columns:
        try:
            df_freq_inv[col] = pd.to_numeric(df_freq_inv[col], errors='coerce')
        except:
            pass
    
    if 'V147' in df_freq_inv.columns:
        freq_freq_inv = df_freq_inv['V147'].value_counts().sort_index()
        freq_freq_inv_pct = df_freq_inv['V147'].value_counts(normalize=True).sort_index() * 100
        
        print(f"\nTotal de casos: {len(df_freq_inv)}")
        print(f"Valores missing: {df_freq_inv['V147'].isna().sum()}")
        print(f"Valores não-missing: {df_freq_inv['V147'].notna().sum()}")
        
        print("\nFrequência absoluta:")
        print(freq_freq_inv.to_string())
        
        print("\nFrequência percentual:")
        for val, count in freq_freq_inv.items():
            pct = freq_freq_inv_pct[val]
            print(f"  {val}: {pct:.2f}%")
        
        # Mostrar também as novas variáveis criadas
        if 'Freq_inv' in df_freq_inv.columns:
            print("\n\n[VARIÁVEL Freq_inv (criada)]")
            print("-"*70)
            freq_freq_inv_var = df_freq_inv['Freq_inv'].value_counts().sort_index()
            freq_freq_inv_var_pct = df_freq_inv['Freq_inv'].value_counts(normalize=True).sort_index() * 100
            
            print(f"Valores missing: {df_freq_inv['Freq_inv'].isna().sum()}")
            print(f"Valores não-missing: {df_freq_inv['Freq_inv'].notna().sum()}")
            
            print("\nFrequência absoluta:")
            print(freq_freq_inv_var.to_string())
            
            print("\nFrequência percentual:")
            for val, count in freq_freq_inv_var.items():
                pct = freq_freq_inv_var_pct[val]
                print(f"  {val}: {pct:.2f}%")
    else:
        print("Variável V147 não encontrada no arquivo com_freq_inv")
        
except Exception as e:
    print(f"Erro ao carregar arquivo com_freq_inv: {e}")

print("\n" + "="*70)

