"""
Script para processar o índice Y_moral em todas as ondas do WVS:
- Padronização em z-score
- Criação do índice Y_moral
- Teste de consistência interna (Alfa de Cronbach)

Ondas e variáveis:
- Wave 2: V307, V308, V309 (já processado)
- Wave 3: V199, V197, V198
- Wave 5: V204, V202, V203
- Wave 6: V204, V203, V203A
- Wave 7: Q184, Q182, Q183
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent

# Configuração das ondas
WAVES_CONFIG = {
    'wave-3-1997-br': {
        'file': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'output_file': 'WV3_Data_Brazil_Csv_v20221107.1_com_zscore.csv',
        'variables': ['V199', 'V197', 'V198']
    },
    'wave-5-2006-br': {
        'file': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'output_file': 'WV5_Data_Brazil_Csv_v20201117.1_com_zscore.csv',
        'variables': ['V204', 'V202', 'V203']
    },
    'wave-6-2014-br': {
        'file': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'output_file': 'WV6_Data_Brazil_Csv_v20221117.1_com_zscore.csv',
        'variables': ['V204', 'V203', 'V203A']
    },
    'wave-7-2018-br': {
        'file': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'output_file': 'WVS_Wave_7_Brazil_Csv_v5.1_com_zscore.csv',
        'variables': ['Q184', 'Q182', 'Q183']
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

def load_wave_data(wave_dir, file_name):
    """Carrega os dados de uma onda."""
    file_path = BASE_DIR / wave_dir / file_name
    
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    try:
        # Tenta ler com ponto-e-vírgula primeiro
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        except:
            # Se falhar, tenta com vírgula
            df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False)
        
        print(f"  Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df
    except Exception as e:
        print(f"  ❌ Erro ao carregar dados: {e}")
        return None

def check_variables(df, variables):
    """Verifica se as variáveis existem no DataFrame."""
    missing_vars = [v for v in variables if v not in df.columns]
    
    if missing_vars:
        print(f"  ⚠️  Variáveis não encontradas: {', '.join(missing_vars)}")
        # Mostra variáveis similares
        print(f"  Variáveis disponíveis que começam com '{variables[0][0]}':")
        similar_vars = [col for col in df.columns if col.startswith(variables[0][0])]
        print(f"    {similar_vars[:10]}")
        return False
    
    return True

def standardize_variables(df, variables):
    """Padroniza as variáveis em z-score."""
    for var in variables:
        z_col = f"{var}_z"
        
        valid_values = df[var].dropna()
        
        if len(valid_values) == 0:
            print(f"  ⚠️  Variável {var} não tem valores válidos. Pulando...")
            continue
        
        mean = valid_values.mean()
        std = valid_values.std()
        
        if std == 0:
            print(f"  ⚠️  Variável {var} tem desvio padrão zero. Pulando...")
            continue
        
        df[z_col] = (df[var] - mean) / std
    
    return df

def create_index(df, variables):
    """Cria o índice Y_moral como média das variáveis padronizadas."""
    z_variables = [f"{v}_z" for v in variables]
    
    # Verifica se todas as variáveis z existem
    missing_z = [v for v in z_variables if v not in df.columns]
    if missing_z:
        print(f"  ⚠️  Variáveis z-score não encontradas: {', '.join(missing_z)}")
        return df
    
    # Calcula a média
    data = df[z_variables].copy()
    df['Y_moral'] = data.mean(axis=1, skipna=False)
    
    return df

def calculate_reliability(df, variables):
    """Calcula o Alfa de Cronbach."""
    df_clean = df[variables].dropna()
    n_valid = len(df_clean)
    
    if n_valid < 10:
        return np.nan, n_valid
    
    alpha = cronbach_alpha(df_clean, variables)
    return alpha, n_valid

def process_wave(wave_name, config):
    """Processa uma onda completa."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()}")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    file_name = config['file']
    variables = config['variables']
    output_file = config['output_file']
    
    # Passo 1: Carregar dados
    print(f"\n[PASSO 1] Carregando dados...")
    df = load_wave_data(wave_dir, file_name)
    if df is None:
        return None
    
    # Verificar variáveis
    if not check_variables(df, variables):
        return None
    
    # Passo 2: Padronização em z-score
    print(f"\n[PASSO 2] Padronizando variáveis em z-score...")
    df = standardize_variables(df, variables)
    
    z_variables = [f"{v}_z" for v in variables]
    for var, z_var in zip(variables, z_variables):
        if z_var in df.columns:
            print(f"  ✓ {var} -> {z_var}: média={df[z_var].mean():.6f}, DP={df[z_var].std():.6f}")
    
    # Passo 3: Criar índice Y_moral
    print(f"\n[PASSO 3] Criando índice Y_moral...")
    df = create_index(df, variables)
    
    if 'Y_moral' in df.columns:
        print(f"  ✓ Y_moral criado: média={df['Y_moral'].mean():.6f}, DP={df['Y_moral'].std():.6f}")
        print(f"  ✓ Valores válidos: {df['Y_moral'].notna().sum()} de {len(df)}")
    
    # Passo 4: Teste de consistência interna
    print(f"\n[PASSO 4] Testando consistência interna (Alfa de Cronbach)...")
    alpha, n_valid = calculate_reliability(df, variables)
    
    if pd.isna(alpha):
        print(f"  ❌ Não foi possível calcular o Alfa de Cronbach")
    else:
        interp = interpret_alpha(alpha)
        print(f"  ✓ Alfa de Cronbach: {alpha:.4f} ({interp})")
        print(f"  ✓ Casos válidos: {n_valid}")
    
    # Salvar arquivo
    print(f"\n[PASSO 5] Salvando arquivo...")
    output_path = BASE_DIR / wave_dir / output_file
    try:
        # Tenta salvar com ponto-e-vírgula (padrão WVS)
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"  ✓ Arquivo salvo: {output_path}")
        print(f"  ✓ Total de colunas: {len(df.columns)}")
    except Exception as e:
        print(f"  ❌ Erro ao salvar: {e}")
        return None
    
    return {
        'wave': wave_name,
        'variables': ', '.join(variables),
        'cronbach_alpha': alpha,
        'interpretacao': interpret_alpha(alpha),
        'n_casos_validos': n_valid,
        'n_total': len(df)
    }

def main():
    """Função principal."""
    print("="*70)
    print("PROCESSAMENTO DO ÍNDICE Y_moral - TODAS AS ONDAS")
    print("="*70)
    
    results = []
    
    # Processa cada onda
    for wave_name, config in WAVES_CONFIG.items():
        result = process_wave(wave_name, config)
        if result:
            results.append(result)
    
    # Salva resumo dos resultados
    if results:
        print(f"\n{'='*70}")
        print("RESUMO DOS RESULTADOS")
        print(f"{'='*70}\n")
        
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        # Salva em CSV
        summary_file = BASE_DIR / 'resumo_y_moral_todas_ondas.csv'
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ Resumo salvo em: {summary_file}")
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()








