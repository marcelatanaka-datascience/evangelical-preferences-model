"""
Script para recalcular o índice Y_institucional usando apenas as 4 variáveis
disponíveis em TODAS as ondas: congresso, justiça, igreja, imprensa.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent

# Variáveis disponíveis em TODAS as ondas
VARS_COMUNS = {
    'congresso': {
        'wave-2-1991-br': 'V279',
        'wave-3-1997-br': 'V144',
        'wave-5-2006-br': 'V140',
        'wave-6-2014-br': 'V117',
        'wave-7-2018-br': 'Q73'
    },
    'justica': {
        'wave-2-1991-br': 'V275',
        'wave-3-1997-br': 'V137',
        'wave-5-2006-br': 'V137',
        'wave-6-2014-br': 'V114',
        'wave-7-2018-br': 'Q70'
    },
    'igreja': {
        'wave-2-1991-br': 'V272',
        'wave-3-1997-br': 'V135',
        'wave-5-2006-br': 'V131',
        'wave-6-2014-br': 'V108',
        'wave-7-2018-br': 'Q64'
    },
    'imprensa': {
        'wave-2-1991-br': 'V276',
        'wave-3-1997-br': 'V138',
        'wave-5-2006-br': 'V133',
        'wave-6-2014-br': 'V110',
        'wave-7-2018-br': 'Q66'
    }
}

WAVES_CONFIG = {
    'wave-2-1991-br': {
        'file_original': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'year': 1991
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'year': 1997
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'year': 2006
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'year': 2014
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'year': 2018
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
    """Processa uma onda usando apenas as 4 variáveis comuns."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()} (4 variáveis comuns)")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    file_original = config['file_original']
    year = config['year']
    
    # Carregar dados
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
        print(f"  ✓ Dados carregados: {len(df)} linhas")
    except Exception as e:
        print(f"  ❌ Erro ao carregar: {e}")
        return None
    
    # Padronizar as 4 variáveis comuns
    print(f"\n[PASSO 2] Padronizando as 4 variáveis comuns...")
    z_variables = {}
    
    for var_name, wave_vars in VARS_COMUNS.items():
        var_col = wave_vars[wave_name]
        
        if var_col not in df.columns:
            print(f"  ⚠️  {var_name} ({var_col}): não encontrada")
            continue
        
        z_col = f"{var_col}_z"
        result = standardize_variable(df, var_col, z_col)
        
        if result:
            z_variables[var_name] = z_col
            valid = df[z_col].dropna()
            print(f"  ✓ {var_name} ({var_col}) -> {z_col}: N={len(valid)}, média={df[z_col].mean():.6f}, DP={df[z_col].std():.6f}")
        else:
            print(f"  ⚠️  {var_name} ({var_col}): não foi possível padronizar")
    
    if len(z_variables) != 4:
        print(f"  ❌ Esperado 4 variáveis, encontradas {len(z_variables)}")
        return None
    
    # Criar índice
    print(f"\n[PASSO 3] Criando índice Y_institucional_4var...")
    z_cols = list(z_variables.values())
    df['Y_institucional_4var'] = df[z_cols].mean(axis=1, skipna=False)
    
    valid_cases = df['Y_institucional_4var'].notna().sum()
    print(f"  ✓ Y_institucional_4var criado: média={df['Y_institucional_4var'].mean():.6f}, DP={df['Y_institucional_4var'].std():.6f}")
    print(f"  ✓ Valores válidos: {valid_cases} de {len(df)}")
    
    # Testar consistência interna
    print(f"\n[PASSO 4] Testando consistência interna...")
    alpha = cronbach_alpha(df, z_cols)
    
    if pd.isna(alpha):
        print(f"  ❌ Não foi possível calcular o Alfa de Cronbach")
    else:
        interp = interpret_alpha(alpha)
        n_valid = df[z_cols].dropna().shape[0]
        print(f"  ✓ Alfa de Cronbach: {alpha:.4f} ({interp})")
        print(f"  ✓ Casos válidos: {n_valid}")
    
    # Preparar DataFrame para empilhamento
    result_df = pd.DataFrame()
    
    for var_name in ['congresso', 'justica', 'igreja', 'imprensa']:
        if var_name in z_variables:
            result_df[f"{var_name}_z"] = df[z_variables[var_name]]
    
    result_df['Y_institucional_4var'] = df['Y_institucional_4var']
    result_df['wave'] = wave_name
    result_df['year'] = year
    result_df['year_dummy'] = year
    
    return {
        'df': result_df,
        'alpha': alpha,
        'n_valid': df[z_cols].dropna().shape[0],
        'wave': wave_name,
        'year': year
    }

def combine_and_test(combined_results):
    """Combina todos os resultados e testa consistência interna."""
    print(f"\n{'='*70}")
    print("EMPILHANDO E TESTANDO CONSISTÊNCIA INTERNA")
    print("="*70)
    
    all_dataframes = [r['df'] for r in combined_results if r is not None]
    
    if not all_dataframes:
        print("❌ Nenhum dado para empilhar.")
        return None
    
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    print(f"\n✓ Total de casos combinados: {len(combined_df)}")
    print(f"✓ Colunas: {', '.join(combined_df.columns)}")
    
    # Testa consistência interna
    z_vars = ['congresso_z', 'justica_z', 'igreja_z', 'imprensa_z']
    
    print(f"\n{'─'*70}")
    print("TESTE DE CONSISTÊNCIA INTERNA - CONJUNTO COMPLETO:")
    print(f"{'─'*70}\n")
    
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
        alpha = cronbach_alpha(wave_df, z_vars)
        n_valid = wave_df[z_vars].dropna().shape[0]
        year = wave_df['year'].iloc[0] if len(wave_df) > 0 else None
        
        interp = interpret_alpha(alpha)
        print(f"{wave_name:20s} ({year}): α = {alpha:.4f} ({interp}), N = {n_valid}")
    
    # Matriz de correlação
    print(f"\n{'─'*70}")
    print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
    print(f"{'─'*70}\n")
    
    corr_matrix = combined_df[z_vars].corr(method='pearson')
    print(corr_matrix.round(4))
    
    # Estatísticas descritivas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(f"{'─'*70}\n")
    
    for var in z_vars + ['Y_institucional_4var']:
        valid = combined_df[var].dropna()
        if len(valid) > 0:
            print(f"{var}:")
            print(f"  N: {len(valid)}, Média: {valid.mean():.6f}, DP: {valid.std():.6f}")
            print(f"  Mín: {valid.min():.6f}, Máx: {valid.max():.6f}")
    
    # Salva dataset
    output_file = BASE_DIR / 'y_institucional_4var_empilhado_completo.csv'
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
    
    summary_data.insert(0, {
        'wave': 'Todas as ondas',
        'year': '1991-2018',
        'cronbach_alpha': alpha_general,
        'interpretacao': interpret_alpha(alpha_general),
        'n_casos_validos': n_valid if not pd.isna(alpha_general) else 0
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = BASE_DIR / 'y_institucional_4var_empilhado_resumo.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ Resumo salvo: {summary_file}")
    
    return combined_df

def main():
    """Função principal."""
    print("="*70)
    print("RECÁLCULO DO ÍNDICE Y_institucional (4 VARIÁVEIS COMUNS)")
    print("Variáveis: congresso, justiça, igreja, imprensa")
    print("="*70)
    
    combined_results = []
    
    for wave_name, config in WAVES_CONFIG.items():
        result = process_wave(wave_name, config)
        if result:
            combined_results.append(result)
    
    if combined_results:
        combined_df = combine_and_test(combined_results)
        
        if combined_df is not None:
            print(f"\n{'='*70}")
            print("PROCESSAMENTO CONCLUÍDO!")
            print("="*70)
            print(f"\nDataset final: {len(combined_df)} casos")
            print(f"Todas as ondas usam as mesmas 4 variáveis!")

if __name__ == "__main__":
    main()








