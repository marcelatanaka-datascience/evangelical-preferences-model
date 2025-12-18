"""
Script para inverter as variáveis do índice Y_moral, padronizar em z-score,
recriar o índice e testar consistência interna.

Passos:
1. Inverter variáveis originais (10 vira 1, 9 vira 2, etc.)
2. Padronizar variáveis invertidas em z-score
3. Recriar índice Y_moral_inv
4. Testar consistência interna
5. Empilhar em dataset único
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
        'file_output': 'WV2_Data_Brazil_Csv_v1.6.1_invertido.csv',
        'year': 1991,
        'variables': {
            'aborto': 'V309',
            'homossex': 'V307',
            'prost': 'V308'
        }
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'file_output': 'WV3_Data_Brazil_Csv_v20221107.1_invertido.csv',
        'year': 1997,
        'variables': {
            'aborto': 'V199',
            'homossex': 'V197',
            'prost': 'V198'
        }
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'file_output': 'WV5_Data_Brazil_Csv_v20201117.1_invertido.csv',
        'year': 2006,
        'variables': {
            'aborto': 'V204',
            'homossex': 'V202',
            'prost': 'V203'
        }
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'file_output': 'WV6_Data_Brazil_Csv_v20221117.1_invertido.csv',
        'year': 2014,
        'variables': {
            'aborto': 'V204',
            'homossex': 'V203',
            'prost': 'V203A'
        }
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'file_output': 'WVS_Wave_7_Brazil_Csv_v5.1_invertido.csv',
        'year': 2018,
        'variables': {
            'aborto': 'Q184',
            'homossex': 'Q182',
            'prost': 'Q183'
        }
    }
}

def cronbach_alpha(df, items):
    """Calcula o Alfa de Cronbach."""
    df_clean = df[items].dropna()
    
    if len(df_clean) < 2:
        return np.nan
    
    k = len(items)
    item_variances = df_clean[items].var()
    total_variance = df_clean[items].sum(axis=1).var()
    
    if total_variance == 0:
        return np.nan
    
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

def interpret_alpha(alpha):
    """Interpreta o valor do alfa."""
    if pd.isna(alpha):
        return "Não calculável"
    elif alpha < 0.5:
        return "Inadequada"
    elif alpha < 0.7:
        return "Aceitável"
    elif alpha < 0.9:
        return "Boa"
    else:
        return "Excelente"

def invert_variable(series):
    """
    Inverte uma variável: 10 vira 1, 9 vira 2, etc.
    Mantém valores missing e valores fora do range 1-10 como estão.
    """
    inverted = series.copy()
    
    # Identifica valores válidos (entre 1 e 10)
    valid_mask = (series >= 1) & (series <= 10)
    
    # Inverte: novo_valor = 11 - valor_original
    inverted[valid_mask] = 11 - series[valid_mask]
    
    return inverted

def process_wave(wave_name, config):
    """Processa uma onda completa: inverte, padroniza, cria índice."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()}")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    file_original = config['file_original']
    file_output = config['file_output']
    variables = config['variables']
    year = config['year']
    
    # Passo 1: Carregar dados originais
    print(f"\n[PASSO 1] Carregando dados originais...")
    file_path = BASE_DIR / wave_dir / file_original
    
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    try:
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        except:
            df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False)
        print(f"  ✓ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
    except Exception as e:
        print(f"  ❌ Erro ao carregar: {e}")
        return None
    
    # Verifica se as variáveis existem
    missing_vars = [v for v in variables.values() if v not in df.columns]
    if missing_vars:
        print(f"  ⚠️  Variáveis não encontradas: {missing_vars}")
        return None
    
    # Passo 2: Inverter variáveis
    print(f"\n[PASSO 2] Invertendo variáveis...")
    for var_name, var_col in variables.items():
        inv_col = f"{var_col}_inv"
        df[inv_col] = invert_variable(df[var_col])
        
        # Estatísticas
        original_valid = df[var_col].dropna()
        inverted_valid = df[inv_col].dropna()
        
        print(f"  ✓ {var_col} -> {inv_col}")
        print(f"    Original: min={original_valid.min()}, max={original_valid.max()}, média={original_valid.mean():.2f}")
        print(f"    Invertido: min={inverted_valid.min()}, max={inverted_valid.max()}, média={inverted_valid.mean():.2f}")
    
    # Passo 3: Padronizar variáveis invertidas em z-score
    print(f"\n[PASSO 3] Padronizando variáveis invertidas em z-score...")
    for var_name, var_col in variables.items():
        inv_col = f"{var_col}_inv"
        z_col = f"{var_col}_inv_z"
        
        valid_values = df[inv_col].dropna()
        
        if len(valid_values) == 0:
            print(f"  ⚠️  {inv_col} não tem valores válidos")
            continue
        
        mean = valid_values.mean()
        std = valid_values.std()
        
        if std == 0:
            print(f"  ⚠️  {inv_col} tem desvio padrão zero")
            continue
        
        df[z_col] = (df[inv_col] - mean) / std
        print(f"  ✓ {inv_col} -> {z_col}: média={df[z_col].mean():.6f}, DP={df[z_col].std():.6f}")
    
    # Passo 4: Criar índice Y_moral_inv
    print(f"\n[PASSO 4] Criando índice Y_moral_inv...")
    z_variables = [f"{v}_inv_z" for v in variables.values()]
    
    # Verifica se todas existem
    missing_z = [v for v in z_variables if v not in df.columns]
    if missing_z:
        print(f"  ⚠️  Variáveis z-score não encontradas: {missing_z}")
        return None
    
    df['Y_moral_inv'] = df[z_variables].mean(axis=1, skipna=False)
    
    print(f"  ✓ Y_moral_inv criado: média={df['Y_moral_inv'].mean():.6f}, DP={df['Y_moral_inv'].std():.6f}")
    print(f"  ✓ Valores válidos: {df['Y_moral_inv'].notna().sum()} de {len(df)}")
    
    # Passo 5: Testar consistência interna
    print(f"\n[PASSO 5] Testando consistência interna...")
    alpha = cronbach_alpha(df, z_variables)
    
    if pd.isna(alpha):
        print(f"  ❌ Não foi possível calcular o Alfa de Cronbach")
    else:
        interp = interpret_alpha(alpha)
        n_valid = df[z_variables].dropna().shape[0]
        print(f"  ✓ Alfa de Cronbach: {alpha:.4f} ({interp})")
        print(f"  ✓ Casos válidos: {n_valid}")
    
    # Salvar arquivo
    print(f"\n[PASSO 6] Salvando arquivo...")
    output_path = BASE_DIR / wave_dir / file_output
    try:
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"  ✓ Arquivo salvo: {output_path}")
        print(f"  ✓ Total de colunas: {len(df.columns)}")
    except Exception as e:
        print(f"  ❌ Erro ao salvar: {e}")
        return None
    
    # Retorna DataFrame com colunas padronizadas para empilhamento
    result_df = pd.DataFrame()
    
    # Adiciona variáveis padronizadas com nomes consistentes
    result_df['aborto_z'] = df[f"{variables['aborto']}_inv_z"]
    result_df['homossex_z'] = df[f"{variables['homossex']}_inv_z"]
    result_df['prost_z'] = df[f"{variables['prost']}_inv_z"]
    result_df['Y_moral_inv'] = df['Y_moral_inv']
    
    # Adiciona informações da wave
    result_df['wave'] = wave_name
    result_df['year'] = year
    result_df['year_dummy'] = year
    
    return {
        'df': result_df,
        'alpha': alpha,
        'n_valid': df[z_variables].dropna().shape[0],
        'wave': wave_name,
        'year': year
    }

def combine_and_test(combined_results):
    """Combina todos os resultados e testa consistência interna."""
    print(f"\n{'='*70}")
    print("EMPILHANDO E TESTANDO CONSISTÊNCIA INTERNA")
    print("="*70)
    
    # Empilha todos os DataFrames
    all_dataframes = [r['df'] for r in combined_results if r is not None]
    
    if not all_dataframes:
        print("❌ Nenhum dado para empilhar.")
        return None
    
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    print(f"\n✓ Total de casos combinados: {len(combined_df)}")
    print(f"✓ Colunas: {', '.join(combined_df.columns)}")
    
    # Testa consistência interna
    print(f"\n{'─'*70}")
    print("TESTE DE CONSISTÊNCIA INTERNA - CONJUNTO COMPLETO:")
    print(f"{'─'*70}\n")
    
    alpha_vars = ['aborto_z', 'homossex_z', 'prost_z']
    alpha_general = cronbach_alpha(combined_df, alpha_vars)
    
    if pd.isna(alpha_general):
        print("❌ Não foi possível calcular o Alfa de Cronbach.")
    else:
        interp = interpret_alpha(alpha_general)
        n_valid = combined_df[alpha_vars].dropna().shape[0]
        print(f"Alfa de Cronbach: {alpha_general:.4f} ({interp})")
        print(f"Casos válidos: {n_valid}")
    
    # Alfa por wave
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH POR ONDA:")
    print(f"{'─'*70}\n")
    
    for wave_name in combined_df['wave'].unique():
        wave_df = combined_df[combined_df['wave'] == wave_name]
        alpha = cronbach_alpha(wave_df, alpha_vars)
        n_valid = wave_df[alpha_vars].dropna().shape[0]
        year = wave_df['year'].iloc[0] if len(wave_df) > 0 else None
        
        interp = interpret_alpha(alpha)
        print(f"{wave_name:20s} ({year}): α = {alpha:.4f} ({interp}), N = {n_valid}")
    
    # Matriz de correlação
    print(f"\n{'─'*70}")
    print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
    print(f"{'─'*70}\n")
    
    corr_matrix = combined_df[alpha_vars].corr(method='pearson')
    print(corr_matrix.round(4))
    
    # Estatísticas descritivas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(f"{'─'*70}\n")
    
    for var in alpha_vars + ['Y_moral_inv']:
        valid = combined_df[var].dropna()
        print(f"{var}:")
        print(f"  N: {len(valid)}, Média: {valid.mean():.6f}, DP: {valid.std():.6f}")
        print(f"  Mín: {valid.min():.6f}, Máx: {valid.max():.6f}")
    
    # Salva dataset empilhado
    output_file = BASE_DIR / 'y_moral_inv_empilhado_completo.csv'
    combined_df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n✓ Dataset empilhado salvo: {output_file}")
    
    # Salva resumo
    summary_data = []
    for r in combined_results:
        if r is not None:
            summary_data.append({
                'wave': r['wave'],
                'year': r['year'],
                'cronbach_alpha': r['alpha'],
                'interpretacao': interpret_alpha(r['alpha']),
                'n_casos_validos': r['n_valid']
            })
    
    # Adiciona resultado geral
    summary_data.insert(0, {
        'wave': 'Todas as ondas',
        'year': '1991-2018',
        'cronbach_alpha': alpha_general,
        'interpretacao': interpret_alpha(alpha_general),
        'n_casos_validos': n_valid
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = BASE_DIR / 'y_moral_inv_empilhado_resumo.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ Resumo salvo: {summary_file}")
    
    return combined_df

def main():
    """Função principal."""
    print("="*70)
    print("INVERSÃO E PROCESSAMENTO DO ÍNDICE Y_moral")
    print("="*70)
    
    # Processa cada onda
    combined_results = []
    
    for wave_name, config in WAVES_CONFIG.items():
        result = process_wave(wave_name, config)
        if result:
            combined_results.append(result)
    
    # Empilha e testa
    if combined_results:
        combined_df = combine_and_test(combined_results)
        
        if combined_df is not None:
            print(f"\n{'='*70}")
            print("PROCESSAMENTO CONCLUÍDO!")
            print("="*70)
            print(f"\nDataset final: {len(combined_df)} casos")
            print(f"Colunas: {', '.join(combined_df.columns)}")

if __name__ == "__main__":
    main()








