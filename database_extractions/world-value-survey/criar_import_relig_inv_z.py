"""
Script para criar as variáveis import_relig_inv e import_relig_inv_z (importância da religião).
Passos:
1. Usar arquivos originais de cada onda
2. Converter todos os valores negativos em missing
3. Inverter os valores originais e criar coluna import_relig_inv (escala 1-4)
4. Padronizar import_relig_inv em z-score e criar coluna import_relig_inv_z
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent

# Configuração das ondas e variáveis
WAVES_CONFIG = {
    'wave-2-1991-br': {
        'file_original': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'file_output': 'WV2_Data_Brazil_Csv_v1.6.1_com_import_relig_inv.csv',
        'year': 1991,
        'variable': 'V9',
        'scale_max': 4  # Escala 1-4
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'file_output': 'WV3_Data_Brazil_Csv_v20221107.1_com_import_relig_inv.csv',
        'year': 1997,
        'variable': 'V9',
        'scale_max': 4  # Escala 1-4
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'file_output': 'WV5_Data_Brazil_Csv_v20201117.1_com_import_relig_inv.csv',
        'year': 2005,
        'variable': 'V9',
        'scale_max': 4  # Escala 1-4
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'file_output': 'WV6_Data_Brazil_Csv_v20221117.1_com_import_relig_inv.csv',
        'year': 2014,
        'variable': 'V9',
        'scale_max': 4  # Escala 1-4
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'file_output': 'WVS_Wave_7_Brazil_Csv_v5.1_com_import_relig_inv.csv',
        'year': 2018,
        'variable': 'Q6',
        'scale_max': 4  # Escala 1-4
    }
}

def process_wave(wave_name, config):
    """Processa uma onda completa."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()}")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    file_original = config['file_original']
    file_output = config['file_output']
    variable = config['variable']
    year = config['year']
    scale_max = config['scale_max']
    
    # Passo 1: Carregar dados originais
    print(f"\n[PASSO 1] Carregando dados originais...")
    file_path = BASE_DIR / wave_dir / file_original
    
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    try:
        # Ler usando csv.reader para lidar com desalinhamento de colunas
        import csv
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader)
            # Ler todas as linhas, garantindo que tenham o mesmo número de colunas do header
            for row in reader:
                # Se a linha tem mais colunas, pegar apenas as primeiras
                if len(row) > len(header):
                    row = row[:len(header)]
                # Se a linha tem menos colunas, preencher com valores vazios
                elif len(row) < len(header):
                    row.extend([''] * (len(header) - len(row)))
                rows.append(row)
        
        # Criar DataFrame
        df = pd.DataFrame(rows, columns=header)
        
        # Converter colunas numéricas
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
        
        print(f"  ✓ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Verifica se a variável existe
        if variable not in df.columns:
            print(f"  ⚠️  Variável {variable} não encontrada nas colunas")
            print(f"  Primeiras colunas: {list(df.columns[:10])}")
            return None
        
        # Mostra estatísticas da variável original
        original_vals = df[variable].dropna()
        negative_count = (df[variable] < 0).sum()
        valid_count = ((df[variable] >= 1) & (df[variable] <= scale_max)).sum()
        
        print(f"  ✓ Variável {variable} encontrada")
        print(f"    Valores únicos: {sorted([int(x) for x in original_vals.unique() if x >= 0])[:15]}")
        print(f"    Valores negativos: {negative_count}")
        print(f"    Valores válidos (1-{scale_max}): {valid_count}")
        if len(original_vals) > 0:
            print(f"    Min: {int(original_vals.min())}, Max: {int(original_vals.max())}")
            
    except Exception as e:
        print(f"  ❌ Erro ao carregar: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Passo 2: Converter valores negativos em missing
    print(f"\n[PASSO 2] Convertendo valores negativos em missing...")
    negative_before = (df[variable] < 0).sum()
    df[variable + '_temp'] = df[variable].copy()
    df.loc[df[variable] < 0, variable + '_temp'] = np.nan
    negative_after = df[variable + '_temp'].isna().sum() - df[variable].isna().sum()
    print(f"  ✓ Valores negativos convertidos: {negative_before} -> {negative_after} missing adicionais")
    
    # Passo 3: Inverter valores e criar import_relig_inv
    print(f"\n[PASSO 3] Invertendo valores e criando import_relig_inv...")
    valid_mask = (df[variable + '_temp'] >= 1) & (df[variable + '_temp'] <= scale_max)
    
    df['import_relig_inv'] = np.nan
    df.loc[valid_mask, 'import_relig_inv'] = (scale_max + 1) - df.loc[valid_mask, variable + '_temp']
    
    # Estatísticas
    original_valid = df[variable + '_temp'].dropna()
    inverted_valid = df['import_relig_inv'].dropna()
    
    print(f"  ✓ {variable} -> import_relig_inv")
    print(f"    Escala: 1-{scale_max} (inversão: {scale_max+1} - valor)")
    if len(original_valid) > 0:
        print(f"    Original válido: min={int(original_valid.min())}, max={int(original_valid.max())}, média={original_valid.mean():.2f}")
    if len(inverted_valid) > 0:
        print(f"    Invertido: min={int(inverted_valid.min())}, max={int(inverted_valid.max())}, média={inverted_valid.mean():.2f}")
        print(f"    Valores válidos: {len(inverted_valid)} de {len(df)}")
    
    # Remove coluna temporária
    df.drop(columns=[variable + '_temp'], inplace=True)
    
    # Passo 4: Padronizar em z-score e criar import_relig_inv_z
    print(f"\n[PASSO 4] Padronizando import_relig_inv em z-score e criando import_relig_inv_z...")
    valid_values = df['import_relig_inv'].dropna()
    
    if len(valid_values) == 0:
        print(f"  ⚠️  Nenhum valor válido para padronizar")
        df['import_relig_inv_z'] = np.nan
    else:
        mean = valid_values.mean()
        std = valid_values.std()
        
        if std == 0:
            print(f"  ⚠️  Desvio padrão zero, não é possível padronizar")
            df['import_relig_inv_z'] = np.nan
        else:
            df['import_relig_inv_z'] = (df['import_relig_inv'] - mean) / std
            
            valid_z = df['import_relig_inv_z'].dropna()
            print(f"  ✓ import_relig_inv -> import_relig_inv_z")
            print(f"    Média: {df['import_relig_inv_z'].mean():.6f}")
            print(f"    DP: {df['import_relig_inv_z'].std():.6f}")
            print(f"    Valores válidos: {len(valid_z)} de {len(df)}")
            if len(valid_z) > 0:
                print(f"    Min: {valid_z.min():.6f}, Max: {valid_z.max():.6f}")
    
    # Salvar arquivo
    print(f"\n[PASSO 5] Salvando arquivo...")
    output_path = BASE_DIR / wave_dir / file_output
    try:
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"  ✓ Arquivo salvo: {output_path}")
        print(f"  ✓ Total de colunas: {len(df.columns)}")
        print(f"  ✓ Novas colunas criadas: import_relig_inv, import_relig_inv_z")
    except Exception as e:
        print(f"  ❌ Erro ao salvar: {e}")
        return None
    
    # Preparar DataFrame para empilhamento
    result_df = pd.DataFrame()
    result_df['import_relig_inv'] = df['import_relig_inv']
    result_df['import_relig_inv_z'] = df['import_relig_inv_z']
    result_df['wave'] = wave_name
    result_df['year'] = year
    
    return result_df

def combine_all_waves(combined_results):
    """Combina todos os resultados em um único dataset."""
    print(f"\n{'='*70}")
    print("EMPILHANDO TODAS AS ONDAS")
    print("="*70)
    
    all_dataframes = [r for r in combined_results if r is not None]
    
    if not all_dataframes:
        print("❌ Nenhum dado para empilhar.")
        return None
    
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    print(f"\n✓ Total de casos combinados: {len(combined_df)}")
    print(f"✓ Colunas: {', '.join(combined_df.columns)}")
    
    # Estatísticas descritivas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS DE import_relig_inv:")
    print(f"{'─'*70}\n")
    
    valid = combined_df['import_relig_inv'].dropna()
    print(f"N válido: {len(valid)} de {len(combined_df)}")
    if len(valid) > 0:
        print(f"Média: {valid.mean():.6f}")
        print(f"DP: {valid.std():.6f}")
        print(f"Mín: {valid.min():.6f}")
        print(f"Máx: {valid.max():.6f}")
    
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS DE import_relig_inv_z:")
    print(f"{'─'*70}\n")
    
    valid_z = combined_df['import_relig_inv_z'].dropna()
    print(f"N válido: {len(valid_z)} de {len(combined_df)}")
    if len(valid_z) > 0:
        print(f"Média: {valid_z.mean():.6f}")
        print(f"DP: {valid_z.std():.6f}")
        print(f"Mín: {valid_z.min():.6f}")
        print(f"Máx: {valid_z.max():.6f}")
    
    # Estatísticas por wave
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DE import_relig_inv POR ONDA:")
    print(f"{'─'*70}\n")
    
    stats_by_wave = combined_df.groupby('wave')['import_relig_inv'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    stats_by_wave.columns = ['N', 'Média', 'DP', 'Mín', 'Máx']
    print(stats_by_wave)
    
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DE import_relig_inv_z POR ONDA:")
    print(f"{'─'*70}\n")
    
    stats_by_wave_z = combined_df.groupby('wave')['import_relig_inv_z'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    stats_by_wave_z.columns = ['N', 'Média', 'DP', 'Mín', 'Máx']
    print(stats_by_wave_z)
    
    # Salva dataset empilhado
    output_file = BASE_DIR / 'import_relig_inv_empilhado_completo.csv'
    combined_df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n✓ Dataset empilhado salvo: {output_file}")
    
    return combined_df

def main():
    """Função principal."""
    print("="*70)
    print("CRIAÇÃO DAS VARIÁVEIS import_relig_inv E import_relig_inv_z")
    print("Importância da religião (invertida e padronizada)")
    print("="*70)
    
    combined_results = []
    
    for wave_name, config in WAVES_CONFIG.items():
        result = process_wave(wave_name, config)
        if result is not None:
            combined_results.append(result)
    
    if combined_results:
        combined_df = combine_all_waves(combined_results)
        
        if combined_df is not None:
            print(f"\n{'='*70}")
            print("PROCESSAMENTO CONCLUÍDO!")
            print("="*70)
            print(f"\nDataset final: {len(combined_df)} casos")
            print(f"Variáveis import_relig_inv e import_relig_inv_z criadas para todas as ondas!")

if __name__ == "__main__":
    main()

