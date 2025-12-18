"""
Script para criar a variável escol_z (escolaridade normalizada em z-score).
Passos:
1. Ler arquivos originais de cada onda
2. Extrair variáveis específicas de escolaridade
3. Converter valores negativos para missing (NaN)
4. Normalizar em z-score (coluna escol_z)
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
        'file_output': 'WV2_Data_Brazil_Csv_v1.6.1_com_escol_z.csv',
        'year': 1991,
        'variable': 'V375'
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'file_output': 'WV3_Data_Brazil_Csv_v20221107.1_com_escol_z.csv',
        'year': 1997,
        'variable': 'V217'
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'file_output': 'WV5_Data_Brazil_Csv_v20201117.1_com_escol_z.csv',
        'year': 2005,
        'variable': 'V238'
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'file_output': 'WV6_Data_Brazil_Csv_v20221117.1_com_escol_z.csv',
        'year': 2014,
        'variable': 'V248'
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'file_output': 'WVS_Wave_7_Brazil_Csv_v5.1_com_escol_z.csv',
        'year': 2018,
        'variable': 'Q275A'
    }
}

def load_wave_data(wave_dir, file_name):
    """
    Carrega dados de uma onda usando leitura direta de CSV para evitar conflitos com indexes.
    """
    file_path = BASE_DIR / wave_dir / file_name
    
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    try:
        # Ler usando csv.reader para lidar com desalinhamento de colunas
        import csv
        rows = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig remove BOM
            reader = csv.reader(f, delimiter=';')
            header = next(reader)
            # Remove aspas e BOM dos nomes das colunas
            header = [col.strip('"').strip('\ufeff') for col in header]
            
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
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        print(f"  ✓ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df
        
    except Exception as e:
        print(f"  ❌ Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_negative_values(series):
    """
    Converte todos os valores negativos para NaN (missing).
    """
    processed = series.copy()
    # Identifica valores negativos e converte para NaN
    processed[processed < 0] = np.nan
    return processed

def standardize_variable(df, var_col, z_col_name):
    """
    Padroniza uma variável em z-score.
    """
    valid_values = df[var_col].dropna()
    
    if len(valid_values) == 0:
        print(f"  ⚠️  Nenhum valor válido para padronizar")
        return False
    
    mean = valid_values.mean()
    std = valid_values.std()
    
    if std == 0:
        print(f"  ⚠️  Desvio padrão zero, não é possível padronizar")
        return False
    
    df[z_col_name] = (df[var_col] - mean) / std
    return True

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
    
    # Passo 1: Carregar dados originais
    print(f"\n[PASSO 1] Carregando dados originais...")
    df = load_wave_data(wave_dir, file_original)
    
    if df is None:
        return None
    
    # Verifica se a variável existe
    if variable not in df.columns:
        # Para Wave 7, tenta Q275 se Q275A não existir
        if wave_name == 'wave-7-2018-br' and variable == 'Q275A':
            if 'Q275' in df.columns:
                print(f"  ⚠️  Variável {variable} não encontrada, usando Q275")
                variable = 'Q275'
                config['variable'] = 'Q275'  # Atualiza a configuração
            else:
                print(f"  ⚠️  Variável {variable} não encontrada nas colunas")
                print(f"  Primeiras colunas: {list(df.columns[:10])}")
                print(f"  Colunas com Q275: {[col for col in df.columns if 'Q275' in col]}")
                return None
        else:
            print(f"  ⚠️  Variável {variable} não encontrada nas colunas")
            print(f"  Primeiras colunas: {list(df.columns[:10])}")
            return None
    
    print(f"  ✓ Variável {variable} encontrada")
    
    # Estatísticas antes do processamento
    original_series = df[variable]
    original_valid = original_series.dropna()
    print(f"  Valores originais: {len(original_valid)} válidos de {len(df)}")
    if len(original_valid) > 0:
        print(f"    Min: {original_valid.min()}, Max: {original_valid.max()}")
        print(f"    Média: {original_valid.mean():.2f}, DP: {original_valid.std():.2f}")
        negative_count = (original_series < 0).sum()
        print(f"    Valores negativos: {negative_count}")
    
    # Passo 2: Converter valores negativos para missing
    print(f"\n[PASSO 2] Convertendo valores negativos para missing...")
    processed_series = process_negative_values(df[variable])
    
    # Atualiza a coluna original com valores processados
    df[variable] = processed_series
    
    # Estatísticas após remoção de negativos
    after_negative_removal = processed_series.dropna()
    print(f"  ✓ Valores negativos convertidos para NaN")
    print(f"  Valores válidos após remoção de negativos: {len(after_negative_removal)}")
    if len(after_negative_removal) > 0:
        print(f"    Min: {after_negative_removal.min()}, Max: {after_negative_removal.max()}")
        print(f"    Média: {after_negative_removal.mean():.2f}, DP: {after_negative_removal.std():.2f}")
    
    # Passo 3: Normalizar em z-score
    print(f"\n[PASSO 3] Normalizando variável em z-score (escol_z)...")
    z_col = 'escol_z'
    result = standardize_variable(df, variable, z_col)
    
    if result:
        valid_z = df[z_col].dropna()
        print(f"  ✓ Variável {variable} -> {z_col}")
        print(f"    Média: {df[z_col].mean():.6f}")
        print(f"    DP: {df[z_col].std():.6f}")
        print(f"    Valores válidos: {len(valid_z)} de {len(df)}")
        if len(valid_z) > 0:
            print(f"    Min: {valid_z.min():.6f}, Max: {valid_z.max():.6f}")
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
        import traceback
        traceback.print_exc()
        return None
    
    # Preparar DataFrame para empilhamento
    result_df = pd.DataFrame()
    result_df['escol_z'] = df[z_col]
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
    print("ESTATÍSTICAS DESCRITIVAS DE escol_z:")
    print(f"{'─'*70}\n")
    
    valid = combined_df['escol_z'].dropna()
    print(f"N válido: {len(valid)} de {len(combined_df)}")
    print(f"Média: {valid.mean():.6f}")
    print(f"DP: {valid.std():.6f}")
    print(f"Mín: {valid.min():.6f}")
    print(f"Máx: {valid.max():.6f}")
    
    # Estatísticas por wave
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DE escol_z POR ONDA:")
    print(f"{'─'*70}\n")
    
    stats_by_wave = combined_df.groupby('wave')['escol_z'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    stats_by_wave.columns = ['N', 'Média', 'DP', 'Mín', 'Máx']
    print(stats_by_wave)
    
    # Salva dataset empilhado
    output_file = BASE_DIR / 'escol_z_empilhado_completo.csv'
    combined_df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n✓ Dataset empilhado salvo: {output_file}")
    
    return combined_df

def main():
    """Função principal."""
    print("="*70)
    print("CRIAÇÃO DA VARIÁVEL escol_z")
    print("Escolaridade normalizada em z-score")
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
            print(f"Variável escol_z criada e padronizada para todas as ondas!")

if __name__ == "__main__":
    main()

