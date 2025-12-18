"""
Script para criar a variável Freq_z (frequência a serviços religiosos).
Passos:
1. Inverter a escala de respostas
2. Padronizar em z-score
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
        'file_output': 'WV2_Data_Brazil_Csv_v1.6.1_com_freq.csv',
        'year': 1991,
        'variable': 'V147',
        'scale_max': 8  # Escala 1-8
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'file_output': 'WV3_Data_Brazil_Csv_v20221107.1_com_freq.csv',
        'year': 1997,
        'variable': 'V181',
        'scale_max': 7  # Escala 1-7
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'file_output': 'WV5_Data_Brazil_Csv_v20201117.1_com_freq.csv',
        'year': 2006,
        'variable': 'V186',
        'scale_max': 7  # Escala 1-7
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'file_output': 'WV6_Data_Brazil_Csv_v20221117.1_com_freq.csv',
        'year': 2014,
        'variable': 'V145',
        'scale_max': 7  # Escala 1-7
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'file_output': 'WVS_Wave_7_Brazil_Csv_v5.1_com_freq.csv',
        'year': 2018,
        'variable': 'Q171',
        'scale_max': 7  # Escala 1-7
    }
}

def invert_variable(series, scale_max):
    """
    Inverte uma variável usando a escala máxima especificada.
    Para escala 1-8: novo_valor = 9 - valor_original
    Para escala 1-7: novo_valor = 8 - valor_original
    Mantém valores missing e valores fora do range como estão.
    """
    inverted = series.copy()
    
    # Identifica valores válidos (>= 1 e <= scale_max)
    valid_mask = (series >= 1) & (series <= scale_max)
    
    # Inverte: novo_valor = (scale_max + 1) - valor_original
    inverted[valid_mask] = (scale_max + 1) - series[valid_mask]
    
    return inverted

def standardize_variable(df, var_col, z_col_name):
    """Padroniza uma variável em z-score."""
    valid_values = df[var_col].dropna()
    
    if len(valid_values) == 0:
        return None
    
    mean = valid_values.mean()
    std = valid_values.std()
    
    if std == 0:
        return None
    
    df[z_col_name] = (df[var_col] - mean) / std
    return True

def process_wave(wave_name, config):
    """Processa uma onda: inverte e padroniza."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()}")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    file_original = config['file_original']
    file_output = config['file_output']
    variable = config['variable']
    year = config['year']
    
    # Passo 1: Carregar dados originais (SEM usar arquivos modificados)
    print(f"\n[PASSO 1] Carregando dados originais...")
    file_path = BASE_DIR / wave_dir / file_original
    
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    try:
        # Tenta primeiro com ponto-e-vírgula
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        except:
            # Se falhar, tenta com vírgula
            try:
                df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False)
            except:
                # Se falhar, tenta sem especificar separador
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        
        print(f"  ✓ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Verifica se a variável existe e mostra valores
        if variable in df.columns:
            valid_vals = df[variable].dropna()
            valid_range = valid_vals[(valid_vals >= 1) & (valid_vals <= 10)]
            print(f"  ✓ Variável {variable} encontrada")
            print(f"    Valores únicos (>=1): {sorted([int(x) for x in valid_range.unique() if x >= 1]) if len(valid_range) > 0 else 'N/A'}")
            if len(valid_range) > 0:
                print(f"    Min: {int(valid_range.min())}, Max: {int(valid_range.max())}")
        else:
            print(f"  ⚠️  Variável {variable} não encontrada nas colunas")
            print(f"  Primeiras colunas: {list(df.columns[:10])}")
    except Exception as e:
        print(f"  ❌ Erro ao carregar: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Verifica se a variável existe
    if variable not in df.columns:
        print(f"  ⚠️  Variável {variable} não encontrada")
        return None
    
    # Passo 2: Inverter variável
    print(f"\n[PASSO 2] Invertendo variável {variable}...")
    inv_col = f"{variable}_inv"
    
    # Usa a escala máxima especificada na configuração
    scale_max = config['scale_max']
    scale_info = f"1-{scale_max} (inversão: {scale_max+1} - valor)"
    
    df[inv_col] = invert_variable(df[variable], scale_max)
    
    # Estatísticas
    original_valid = df[variable].dropna()
    inverted_valid = df[inv_col].dropna()
    
    print(f"  ✓ {variable} -> {inv_col}")
    print(f"    Escala usada: {scale_info}")
    print(f"    Original: min={original_valid.min()}, max={original_valid.max()}, média={original_valid.mean():.2f}")
    print(f"    Invertido: min={inverted_valid.min()}, max={inverted_valid.max()}, média={inverted_valid.mean():.2f}")
    
    # Passo 3: Padronizar em z-score
    print(f"\n[PASSO 3] Padronizando variável invertida em z-score...")
    z_col = 'Freq_z'
    result = standardize_variable(df, inv_col, z_col)
    
    if result:
        valid = df[z_col].dropna()
        print(f"  ✓ {inv_col} -> {z_col}")
        print(f"    Média: {df[z_col].mean():.6f}")
        print(f"    DP: {df[z_col].std():.6f}")
        print(f"    Valores válidos: {len(valid)} de {len(df)}")
    else:
        print(f"  ⚠️  Não foi possível padronizar")
        return None
    
    # Salvar arquivo
    print(f"\n[PASSO 4] Salvando arquivo...")
    output_path = BASE_DIR / wave_dir / file_output
    try:
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"  ✓ Arquivo salvo: {output_path}")
        print(f"  ✓ Total de colunas: {len(df.columns)}")
    except Exception as e:
        print(f"  ❌ Erro ao salvar: {e}")
        return None
    
    # Preparar DataFrame para empilhamento
    result_df = pd.DataFrame()
    result_df['Freq_z'] = df[z_col]
    result_df['wave'] = wave_name
    result_df['year'] = year
    result_df['year_dummy'] = year
    
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
    print("ESTATÍSTICAS DESCRITIVAS DE Freq_z:")
    print(f"{'─'*70}\n")
    
    valid = combined_df['Freq_z'].dropna()
    print(f"N válido: {len(valid)} de {len(combined_df)}")
    print(f"Média: {valid.mean():.6f}")
    print(f"DP: {valid.std():.6f}")
    print(f"Mín: {valid.min():.6f}")
    print(f"Máx: {valid.max():.6f}")
    
    # Estatísticas por wave
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DE Freq_z POR ONDA:")
    print(f"{'─'*70}\n")
    
    stats_by_wave = combined_df.groupby('wave')['Freq_z'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    stats_by_wave.columns = ['N', 'Média', 'DP', 'Mín', 'Máx']
    print(stats_by_wave)
    
    # Salva dataset empilhado
    output_file = BASE_DIR / 'freq_z_empilhado_completo.csv'
    combined_df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n✓ Dataset empilhado salvo: {output_file}")
    
    return combined_df

def main():
    """Função principal."""
    print("="*70)
    print("CRIAÇÃO DA VARIÁVEL Freq_z")
    print("Frequência a serviços religiosos (invertida e padronizada)")
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
            print(f"Variável Freq_z criada e padronizada para todas as ondas!")

if __name__ == "__main__":
    main()

