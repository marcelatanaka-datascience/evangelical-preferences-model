"""
Script para empilhar as variáveis padronizadas em z-score de todas as ondas
em um único dataset e testar a consistência interna do índice Y_moral.

Estrutura das variáveis:
- aborto_z: wave 2 V309_z, wave 3 V199_z, Wave 5 V204_z, wave 6 V204_z, wave 7 Q184_z
- homossex_z: Wave 2 V307_z, Wave 3 V197_z, wave 5 V202_z, Wave 6 V203_z, wave 7 Q182_z
- prost_z: Wave 2 V308_z, wave 3 V198_z, Wave 5 V203_z, wave 6 V203A_z, Wave 7 Q183_z
- Y_moral: índice já calculado
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent

# Configuração das ondas e mapeamento de variáveis
WAVES_CONFIG = {
    'wave-2-1991-br': {
        'file': 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv',
        'year': 1991,
        'aborto': 'V309_z',
        'homossex': 'V307_z',
        'prost': 'V308_z'
    },
    'wave-3-1997-br': {
        'file': 'WV3_Data_Brazil_Csv_v20221107.1_com_zscore.csv',
        'year': 1997,
        'aborto': 'V199_z',
        'homossex': 'V197_z',
        'prost': 'V198_z'
    },
    'wave-5-2006-br': {
        'file': 'WV5_Data_Brazil_Csv_v20201117.1_com_zscore.csv',
        'year': 2006,
        'aborto': 'V204_z',
        'homossex': 'V202_z',
        'prost': 'V203_z'
    },
    'wave-6-2014-br': {
        'file': 'WV6_Data_Brazil_Csv_v20221117.1_com_zscore.csv',
        'year': 2014,
        'aborto': 'V204_z',
        'homossex': 'V203_z',
        'prost': 'V203A_z'
    },
    'wave-7-2018-br': {
        'file': 'WVS_Wave_7_Brazil_Csv_v5.1_com_zscore.csv',
        'year': 2018,
        'aborto': 'Q184_z',
        'homossex': 'Q182_z',
        'prost': 'Q183_z'
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

def load_and_standardize_wave(wave_name, config):
    """Carrega uma onda e padroniza as variáveis com nomes consistentes."""
    file_path = BASE_DIR / wave_name / config['file']
    
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
        
        # Verifica se as variáveis existem
        missing_vars = []
        for var_name, var_col in [('aborto', config['aborto']), 
                                   ('homossex', config['homossex']), 
                                   ('prost', config['prost'])]:
            if var_col not in df.columns:
                missing_vars.append(f"{var_name} ({var_col})")
        
        if missing_vars:
            print(f"⚠️  {wave_name}: Variáveis não encontradas: {', '.join(missing_vars)}")
            return None
        
        # Cria um novo DataFrame com as colunas padronizadas
        result_df = pd.DataFrame()
        
        # Adiciona variáveis padronizadas com nomes consistentes
        result_df['aborto_z'] = df[config['aborto']]
        result_df['homossex_z'] = df[config['homossex']]
        result_df['prost_z'] = df[config['prost']]
        
        # Adiciona Y_moral se existir
        if 'Y_moral' in df.columns:
            result_df['Y_moral'] = df['Y_moral']
        else:
            # Calcula Y_moral como média das 3 variáveis
            result_df['Y_moral'] = result_df[['aborto_z', 'homossex_z', 'prost_z']].mean(axis=1)
        
        # Adiciona informações da wave
        result_df['wave'] = wave_name
        result_df['year'] = config['year']
        
        # Adiciona variável dummy para o ano
        result_df['year_dummy'] = config['year']
        
        print(f"  ✓ {wave_name}: {len(result_df)} casos processados")
        return result_df
        
    except Exception as e:
        print(f"  ❌ Erro ao processar {wave_name}: {e}")
        return None

def combine_all_waves():
    """Combina todas as ondas em um único dataset."""
    print("="*70)
    print("EMPILHANDO VARIÁVEIS DE TODAS AS ONDAS")
    print("="*70)
    
    all_dataframes = []
    
    for wave_name, config in WAVES_CONFIG.items():
        print(f"\nProcessando {wave_name}...")
        df = load_and_standardize_wave(wave_name, config)
        if df is not None:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("\n❌ Nenhum dado foi processado.")
        return None
    
    # Empilha todos os DataFrames
    print(f"\n{'─'*70}")
    print("Empilhando DataFrames...")
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    print(f"✓ Total de casos combinados: {len(combined_df)}")
    print(f"✓ Colunas: {', '.join(combined_df.columns)}")
    
    return combined_df

def test_reliability(combined_df):
    """Testa a consistência interna do índice Y_moral."""
    print(f"\n{'='*70}")
    print("TESTE DE CONSISTÊNCIA INTERNA DO ÍNDICE Y_moral")
    print("="*70)
    
    # Verifica variáveis
    required_vars = ['aborto_z', 'homossex_z', 'prost_z', 'Y_moral']
    missing_vars = [v for v in required_vars if v not in combined_df.columns]
    
    if missing_vars:
        print(f"❌ Variáveis não encontradas: {', '.join(missing_vars)}")
        return None
    
    # Estatísticas gerais
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS GERAIS:")
    print(f"{'─'*70}\n")
    
    print(f"Total de casos: {len(combined_df)}")
    print(f"Casos válidos (todas as variáveis): {combined_df[required_vars].dropna().shape[0]}")
    
    # Estatísticas descritivas das variáveis padronizadas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS PADRONIZADAS:")
    print(f"{'─'*70}\n")
    
    for var in ['aborto_z', 'homossex_z', 'prost_z', 'Y_moral']:
        valid = combined_df[var].dropna()
        print(f"{var}:")
        print(f"  N válido: {len(valid)}")
        print(f"  Média: {valid.mean():.6f}")
        print(f"  DP: {valid.std():.6f}")
        print(f"  Mín: {valid.min():.6f}")
        print(f"  Máx: {valid.max():.6f}")
    
    # Matriz de correlação
    print(f"\n{'─'*70}")
    print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
    print(f"{'─'*70}\n")
    
    corr_matrix = combined_df[['aborto_z', 'homossex_z', 'prost_z']].corr(method='pearson')
    print(corr_matrix.round(4))
    
    # Alfa de Cronbach - Geral
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH - CONJUNTO COMPLETO:")
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
        print(f"Variáveis: {', '.join(alpha_vars)}")
    
    # Alfa de Cronbach por wave
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH POR ONDA:")
    print(f"{'─'*70}\n")
    
    results_by_wave = []
    
    for wave_name in combined_df['wave'].unique():
        wave_df = combined_df[combined_df['wave'] == wave_name]
        alpha = cronbach_alpha(wave_df, alpha_vars)
        n_valid = wave_df[alpha_vars].dropna().shape[0]
        year = wave_df['year'].iloc[0] if len(wave_df) > 0 else None
        
        interp = interpret_alpha(alpha)
        print(f"{wave_name:20s} ({year}): α = {alpha:.4f} ({interp}), N = {n_valid}")
        
        results_by_wave.append({
            'wave': wave_name,
            'year': year,
            'cronbach_alpha': alpha,
            'interpretacao': interp,
            'n_casos_validos': n_valid
        })
    
    # Estatísticas por ano
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DO ÍNDICE Y_moral POR ANO:")
    print(f"{'─'*70}\n")
    
    stats_by_year = combined_df.groupby('year')['Y_moral'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    stats_by_year.columns = ['N', 'Média', 'DP', 'Mín', 'Máx']
    print(stats_by_year)
    
    return {
        'alpha_general': alpha_general,
        'results_by_wave': results_by_wave,
        'stats_by_year': stats_by_year
    }

def save_results(combined_df, results):
    """Salva os resultados."""
    print(f"\n{'='*70}")
    print("SALVANDO RESULTADOS")
    print("="*70)
    
    # Salva dataset combinado
    output_file = BASE_DIR / 'y_moral_empilhado_completo.csv'
    combined_df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"✓ Dataset empilhado salvo: {output_file}")
    print(f"  Total de casos: {len(combined_df)}")
    print(f"  Total de colunas: {len(combined_df.columns)}")
    print(f"  Colunas: {', '.join(combined_df.columns)}")
    
    # Salva resultados do teste
    if results:
        results_df = pd.DataFrame(results['results_by_wave'])
        results_file = BASE_DIR / 'y_moral_empilhado_resultados.csv'
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"✓ Resultados salvos: {results_file}")
        
        # Adiciona resultado geral
        summary = pd.DataFrame([{
            'wave': 'Todas as ondas',
            'year': '1991-2018',
            'cronbach_alpha': results['alpha_general'],
            'interpretacao': interpret_alpha(results['alpha_general']),
            'n_casos_validos': combined_df[['aborto_z', 'homossex_z', 'prost_z']].dropna().shape[0]
        }])
        
        summary_file = BASE_DIR / 'y_moral_empilhado_resumo.csv'
        summary_df = pd.concat([summary, results_df], ignore_index=True)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✓ Resumo completo salvo: {summary_file}")

def main():
    """Função principal."""
    print("="*70)
    print("EMPILHAMENTO E TESTE DE CONSISTÊNCIA INTERNA - Y_moral")
    print("="*70)
    
    # Empilha todas as ondas
    combined_df = combine_all_waves()
    if combined_df is None:
        return
    
    # Testa consistência interna
    results = test_reliability(combined_df)
    
    # Salva resultados
    save_results(combined_df, results)
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*70)

if __name__ == "__main__":
    main()








