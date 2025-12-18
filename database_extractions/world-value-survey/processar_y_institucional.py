"""
Script para criar o índice Y_institucional a partir de 7 temas de confiança institucional.

Temas:
1. Apoio à democracia
2. Confiança - Governo Federal
3. Confiança - Partidos Políticos
4. Confiança - Congresso
5. Confiança - Justiça
6. Confiança - Igreja
7. Confiança - Imprensa

Passos:
1. Padronizar variáveis em z-score (SEM inversão)
2. Criar índice Y_institucional por onda
3. Testar consistência interna
4. Empilhar em dataset único
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
        'file_output': 'WV2_Data_Brazil_Csv_v1.6.1_institucional.csv',
        'year': 1991,
        'variables': {
            'democracia': None,  # N/A
            'governo_federal': None,  # N/A
            'partidos': None,  # N/A
            'congresso': 'V279',
            'justica': 'V275',
            'igreja': 'V272',
            'imprensa': 'V276'
        }
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'file_output': 'WV3_Data_Brazil_Csv_v20221107.1_institucional.csv',
        'year': 1997,
        'variables': {
            'democracia': 'V157',
            'governo_federal': 'V142',
            'partidos': 'V143',
            'congresso': 'V144',
            'justica': 'V137',
            'igreja': 'V135',
            'imprensa': 'V138'
        }
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'file_output': 'WV5_Data_Brazil_Csv_v20201117.1_institucional.csv',
        'year': 2006,
        'variables': {
            'democracia': 'V151',
            'governo_federal': 'V138',
            'partidos': 'V139',
            'congresso': 'V140',
            'justica': 'V137',
            'igreja': 'V131',
            'imprensa': 'V133'
        }
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'file_output': 'WV6_Data_Brazil_Csv_v20221117.1_institucional.csv',
        'year': 2014,
        'variables': {
            'democracia': 'V130',
            'governo_federal': 'V115',
            'partidos': 'V116',
            'congresso': 'V117',
            'justica': 'V114',
            'igreja': 'V108',
            'imprensa': 'V110'
        }
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'file_output': 'WVS_Wave_7_Brazil_Csv_v5.1_institucional.csv',
        'year': 2018,
        'variables': {
            'democracia': 'Q238',
            'governo_federal': 'Q238',  # Nota: parece ser a mesma variável, pode precisar verificar
            'partidos': 'Q72',
            'congresso': 'Q73',
            'justica': 'Q70',
            'igreja': 'Q64',
            'imprensa': 'Q66'
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
    """Processa uma onda completa: padroniza, cria índice."""
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
    
    # Verifica quais variáveis existem
    available_vars = {}
    missing_vars = []
    
    for var_name, var_col in variables.items():
        if var_col is None:
            print(f"  ⚠️  {var_name}: N/A nesta onda")
            continue
        elif var_col not in df.columns:
            missing_vars.append(f"{var_name} ({var_col})")
        else:
            available_vars[var_name] = var_col
    
    if missing_vars:
        print(f"  ⚠️  Variáveis não encontradas: {', '.join(missing_vars)}")
    
    if not available_vars:
        print(f"  ❌ Nenhuma variável disponível para esta onda")
        return None
    
    print(f"  ✓ Variáveis disponíveis: {len(available_vars)} de {len([v for v in variables.values() if v is not None])}")
    
    # Passo 2: Padronizar variáveis em z-score
    print(f"\n[PASSO 2] Padronizando variáveis em z-score...")
    z_variables = {}
    
    for var_name, var_col in available_vars.items():
        z_col = f"{var_col}_z"
        result = standardize_variable(df, var_col, z_col)
        
        if result:
            z_variables[var_name] = z_col
            valid = df[z_col].dropna()
            print(f"  ✓ {var_col} -> {z_col}: média={df[z_col].mean():.6f}, DP={df[z_col].std():.6f}, N={len(valid)}")
        else:
            print(f"  ⚠️  {var_col}: não foi possível padronizar")
    
    if not z_variables:
        print(f"  ❌ Nenhuma variável foi padronizada")
        return None
    
    # Passo 3: Criar índice Y_institucional
    print(f"\n[PASSO 3] Criando índice Y_institucional...")
    z_cols = list(z_variables.values())
    
    df['Y_institucional'] = df[z_cols].mean(axis=1, skipna=False)
    
    valid_cases = df['Y_institucional'].notna().sum()
    print(f"  ✓ Y_institucional criado: média={df['Y_institucional'].mean():.6f}, DP={df['Y_institucional'].std():.6f}")
    print(f"  ✓ Valores válidos: {valid_cases} de {len(df)}")
    print(f"  ✓ Variáveis utilizadas: {', '.join(z_variables.keys())}")
    
    # Passo 4: Testar consistência interna
    print(f"\n[PASSO 4] Testando consistência interna...")
    alpha = cronbach_alpha(df, z_cols)
    
    if pd.isna(alpha):
        print(f"  ❌ Não foi possível calcular o Alfa de Cronbach")
    else:
        interp = interpret_alpha(alpha)
        n_valid = df[z_cols].dropna().shape[0]
        print(f"  ✓ Alfa de Cronbach: {alpha:.4f} ({interp})")
        print(f"  ✓ Casos válidos: {n_valid}")
    
    # Salvar arquivo
    print(f"\n[PASSO 5] Salvando arquivo...")
    output_path = BASE_DIR / wave_dir / file_output
    try:
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"  ✓ Arquivo salvo: {output_path}")
        print(f"  ✓ Total de colunas: {len(df.columns)}")
    except Exception as e:
        print(f"  ❌ Erro ao salvar: {e}")
        return None
    
    # Prepara DataFrame para empilhamento com nomes consistentes
    result_df = pd.DataFrame()
    
    # Mapeia variáveis para nomes consistentes
    var_mapping = {
        'democracia': 'democracia_z',
        'governo_federal': 'governo_federal_z',
        'partidos': 'partidos_z',
        'congresso': 'congresso_z',
        'justica': 'justica_z',
        'igreja': 'igreja_z',
        'imprensa': 'imprensa_z'
    }
    
    for var_name, standard_name in var_mapping.items():
        if var_name in z_variables:
            result_df[standard_name] = df[z_variables[var_name]]
    
    result_df['Y_institucional'] = df['Y_institucional']
    result_df['wave'] = wave_name
    result_df['year'] = year
    result_df['year_dummy'] = year
    
    return {
        'df': result_df,
        'alpha': alpha,
        'n_valid': df[z_cols].dropna().shape[0],
        'wave': wave_name,
        'year': year,
        'variables_used': list(z_variables.keys())
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
    
    # Identifica todas as variáveis z disponíveis
    z_vars = [col for col in combined_df.columns if col.endswith('_z')]
    print(f"✓ Variáveis padronizadas disponíveis: {', '.join(z_vars)}")
    
    # Testa consistência interna
    print(f"\n{'─'*70}")
    print("TESTE DE CONSISTÊNCIA INTERNA - CONJUNTO COMPLETO:")
    print(f"{'─'*70}\n")
    
    if not z_vars:
        print("❌ Nenhuma variável padronizada encontrada.")
        return None
    
    alpha_general = cronbach_alpha(combined_df, z_vars)
    
    if pd.isna(alpha_general):
        print("❌ Não foi possível calcular o Alfa de Cronbach.")
    else:
        interp = interpret_alpha(alpha_general)
        n_valid = combined_df[z_vars].dropna().shape[0]
        print(f"Alfa de Cronbach: {alpha_general:.4f} ({interp})")
        print(f"Casos válidos: {n_valid}")
        print(f"Variáveis: {', '.join(z_vars)}")
    
    # Alfa por wave
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH POR ONDA:")
    print(f"{'─'*70}\n")
    
    for wave_name in combined_df['wave'].unique():
        wave_df = combined_df[combined_df['wave'] == wave_name]
        wave_z_vars = [col for col in wave_df.columns if col.endswith('_z')]
        
        if not wave_z_vars:
            print(f"{wave_name:20s}: Nenhuma variável disponível")
            continue
        
        alpha = cronbach_alpha(wave_df, wave_z_vars)
        n_valid = wave_df[wave_z_vars].dropna().shape[0]
        year = wave_df['year'].iloc[0] if len(wave_df) > 0 else None
        
        interp = interpret_alpha(alpha)
        print(f"{wave_name:20s} ({year}): α = {alpha:.4f} ({interp}), N = {n_valid}, Vars = {len(wave_z_vars)}")
    
    # Matriz de correlação (se houver variáveis suficientes)
    if len(z_vars) > 1:
        print(f"\n{'─'*70}")
        print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
        print(f"{'─'*70}\n")
        
        corr_matrix = combined_df[z_vars].corr(method='pearson')
        print(corr_matrix.round(4))
    
    # Estatísticas descritivas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(f"{'─'*70}\n")
    
    for var in z_vars + ['Y_institucional']:
        valid = combined_df[var].dropna()
        if len(valid) > 0:
            print(f"{var}:")
            print(f"  N: {len(valid)}, Média: {valid.mean():.6f}, DP: {valid.std():.6f}")
            print(f"  Mín: {valid.min():.6f}, Máx: {valid.max():.6f}")
    
    # Salva dataset empilhado
    output_file = BASE_DIR / 'y_institucional_empilhado_completo.csv'
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
                'n_casos_validos': r['n_valid'],
                'n_variaveis': len(r['variables_used'])
            })
    
    # Adiciona resultado geral
    summary_data.insert(0, {
        'wave': 'Todas as ondas',
        'year': '1991-2018',
        'cronbach_alpha': alpha_general,
        'interpretacao': interpret_alpha(alpha_general),
        'n_casos_validos': n_valid if not pd.isna(alpha_general) else 0,
        'n_variaveis': len(z_vars)
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = BASE_DIR / 'y_institucional_empilhado_resumo.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ Resumo salvo: {summary_file}")
    
    return combined_df

def main():
    """Função principal."""
    print("="*70)
    print("CRIAÇÃO DO ÍNDICE Y_institucional")
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








