"""
Script para testar o índice Y_moral de maneira longitudinal,
combinando todas as ondas do World Values Survey.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent

# Configuração das ondas e variáveis
WAVES_CONFIG = {
    'wave-2-1991-br': {
        'file': 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv',
        'variables': ['V307', 'V308', 'V309'],
        'year': 1991
    },
    'wave-3-1997-br': {
        'file': 'WV3_Data_Brazil_Csv_v20221107.1_com_zscore.csv',
        'variables': ['V199', 'V197', 'V198'],
        'year': 1997
    },
    'wave-5-2006-br': {
        'file': 'WV5_Data_Brazil_Csv_v20201117.1_com_zscore.csv',
        'variables': ['V204', 'V202', 'V203'],
        'year': 2006
    },
    'wave-6-2014-br': {
        'file': 'WV6_Data_Brazil_Csv_v20221117.1_com_zscore.csv',
        'variables': ['V204', 'V203', 'V203A'],
        'year': 2014
    },
    'wave-7-2018-br': {
        'file': 'WVS_Wave_7_Brazil_Csv_v5.1_com_zscore.csv',
        'variables': ['Q184', 'Q182', 'Q183'],
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

def load_wave_data(wave_name, config):
    """Carrega os dados de uma onda e adiciona informações da wave."""
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
        
        # Adiciona coluna identificando a wave
        df['wave'] = wave_name
        df['year'] = config['year']
        
        print(f"  ✓ {wave_name}: {len(df)} casos")
        return df
    except Exception as e:
        print(f"  ❌ Erro ao carregar {wave_name}: {e}")
        return None

def combine_all_waves():
    """Combina todos os dados de todas as ondas."""
    print("="*70)
    print("COMBINANDO DADOS DE TODAS AS ONDAS")
    print("="*70)
    
    all_dataframes = []
    
    for wave_name, config in WAVES_CONFIG.items():
        print(f"\nCarregando {wave_name}...")
        df = load_wave_data(wave_name, config)
        if df is not None:
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("\n❌ Nenhum dado foi carregado.")
        return None
    
    # Combina todos os DataFrames
    print(f"\n{'─'*70}")
    print("Combinando DataFrames...")
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    print(f"✓ Total de casos combinados: {len(combined_df)}")
    print(f"✓ Total de colunas: {len(combined_df.columns)}")
    
    return combined_df

def test_longitudinal_reliability(combined_df):
    """Testa a consistência interna do índice Y_moral no conjunto combinado."""
    print(f"\n{'='*70}")
    print("TESTE DE CONSISTÊNCIA INTERNA LONGITUDINAL")
    print("="*70)
    
    # Verifica se Y_moral existe
    if 'Y_moral' not in combined_df.columns:
        print("❌ Coluna Y_moral não encontrada no dataset combinado.")
        return None
    
    # Estatísticas gerais
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS GERAIS DO ÍNDICE Y_moral (TODAS AS ONDAS):")
    print(f"{'─'*70}\n")
    
    valid_cases = combined_df['Y_moral'].notna().sum()
    total_cases = len(combined_df)
    
    print(f"Total de casos: {total_cases}")
    print(f"Casos válidos: {valid_cases}")
    print(f"Casos missing: {total_cases - valid_cases}")
    print(f"\nEstatísticas descritivas:")
    print(combined_df['Y_moral'].describe())
    
    # Estatísticas por wave
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DO ÍNDICE Y_moral POR ONDA:")
    print(f"{'─'*70}\n")
    
    stats_by_wave = combined_df.groupby('wave')['Y_moral'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    stats_by_wave.columns = ['N', 'Média', 'DP', 'Mín', 'Máx']
    print(stats_by_wave)
    
    # Teste de Alfa de Cronbach por wave
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH POR ONDA:")
    print(f"{'─'*70}\n")
    
    results_by_wave = []
    
    for wave_name, config in WAVES_CONFIG.items():
        wave_df = combined_df[combined_df['wave'] == wave_name]
        variables = config['variables']
        
        # Verifica se as variáveis existem
        missing_vars = [v for v in variables if v not in wave_df.columns]
        if missing_vars:
            print(f"{wave_name}: Variáveis não encontradas: {missing_vars}")
            continue
        
        alpha = cronbach_alpha(wave_df, variables)
        n_valid = wave_df[variables].dropna().shape[0]
        
        interp = interpret_alpha(alpha)
        print(f"{wave_name:20s}: α = {alpha:.4f} ({interp}), N = {n_valid}")
        
        results_by_wave.append({
            'wave': wave_name,
            'year': config['year'],
            'variables': ', '.join(variables),
            'cronbach_alpha': alpha,
            'interpretacao': interp,
            'n_casos_validos': n_valid
        })
    
    # Teste de Alfa de Cronbach no conjunto completo
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH - CONJUNTO COMPLETO (TODAS AS ONDAS):")
    print(f"{'─'*70}\n")
    
    # Para o conjunto completo, precisamos padronizar as variáveis novamente
    # porque cada wave tem variáveis diferentes
    # Vamos usar as variáveis padronizadas (z-scores) que já existem
    
    # Identifica todas as variáveis z disponíveis
    z_vars = [col for col in combined_df.columns if col.endswith('_z')]
    
    # Como cada wave tem variáveis diferentes, vamos criar um índice padronizado
    # usando apenas Y_moral que já foi calculado por wave
    
    # Mas para o teste de consistência interna, precisamos das variáveis originais
    # Vamos fazer uma abordagem diferente: testar se Y_moral é consistente entre waves
    
    # Alternativa: calcular correlação entre Y_moral de diferentes waves
    # ou fazer análise de variância
    
    print("Nota: Como cada wave tem variáveis diferentes,")
    print("o teste de Alfa de Cronbach no conjunto completo")
    print("requer uma abordagem diferente.")
    print("\nVamos analisar a consistência do índice Y_moral:")
    
    # Análise de variância entre waves
    print(f"\n{'─'*70}")
    print("ANÁLISE DE VARIÂNCIA ENTRE ONDAS:")
    print(f"{'─'*70}\n")
    
    from scipy import stats as scipy_stats
    
    # Teste F (ANOVA)
    groups = [combined_df[combined_df['wave'] == wave]['Y_moral'].dropna() 
              for wave in combined_df['wave'].unique()]
    
    if len(groups) > 1:
        f_stat, p_value = scipy_stats.f_oneway(*groups)
        print(f"Teste F (ANOVA):")
        print(f"  F = {f_stat:.4f}")
        print(f"  p-value = {p_value:.6f}")
        
        if p_value < 0.001:
            sig = "altamente significativa (p < 0.001)"
        elif p_value < 0.01:
            sig = "muito significativa (p < 0.01)"
        elif p_value < 0.05:
            sig = "significativa (p < 0.05)"
        else:
            sig = "não significativa (p >= 0.05)"
        
        print(f"  Diferença entre ondas: {sig}")
    
    # Correlação entre Y_moral e year (tendência temporal)
    print(f"\n{'─'*70}")
    print("TENDÊNCIA TEMPORAL:")
    print(f"{'─'*70}\n")
    
    # Média de Y_moral por ano
    mean_by_year = combined_df.groupby('year')['Y_moral'].mean().sort_index()
    print("Média de Y_moral por ano:")
    for year, mean_val in mean_by_year.items():
        print(f"  {year}: {mean_val:.4f}")
    
    # Correlação entre Y_moral e year
    valid_data = combined_df[['Y_moral', 'year']].dropna()
    if len(valid_data) > 1:
        corr = valid_data['Y_moral'].corr(valid_data['year'])
        print(f"\nCorrelação Y_moral vs Ano: r = {corr:.4f}")
    
    # Salva resultados
    results_df = pd.DataFrame(results_by_wave)
    
    # Adiciona resultado do conjunto completo (se possível)
    summary = {
        'wave': 'Todas as ondas',
        'year': '1991-2018',
        'variables': 'Variáveis específicas por wave',
        'cronbach_alpha': np.nan,  # Não aplicável
        'interpretacao': 'N/A',
        'n_casos_validos': valid_cases,
        'f_statistic': f_stat if len(groups) > 1 else np.nan,
        'p_value': p_value if len(groups) > 1 else np.nan,
        'correlacao_temporal': corr if len(valid_data) > 1 else np.nan
    }
    
    return results_df, summary

def save_results(combined_df, results_by_wave, summary):
    """Salva os resultados e o dataset combinado."""
    print(f"\n{'='*70}")
    print("SALVANDO RESULTADOS")
    print("="*70)
    
    # Salva dataset combinado
    output_file = BASE_DIR / 'y_moral_longitudinal_completo.csv'
    combined_df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"✓ Dataset combinado salvo: {output_file}")
    print(f"  Total de casos: {len(combined_df)}")
    print(f"  Total de colunas: {len(combined_df.columns)}")
    
    # Salva resultados por wave
    results_file = BASE_DIR / 'y_moral_longitudinal_resultados.csv'
    results_by_wave.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"✓ Resultados por wave salvos: {results_file}")
    
    # Salva resumo
    summary_df = pd.DataFrame([summary])
    summary_file = BASE_DIR / 'y_moral_longitudinal_resumo.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ Resumo longitudinal salvo: {summary_file}")

def main():
    """Função principal."""
    print("="*70)
    print("TESTE LONGITUDINAL DO ÍNDICE Y_moral")
    print("="*70)
    
    # Combina todas as ondas
    combined_df = combine_all_waves()
    if combined_df is None:
        return
    
    # Testa consistência interna longitudinal
    results = test_longitudinal_reliability(combined_df)
    if results is None:
        return
    
    results_by_wave, summary = results
    
    # Salva resultados
    save_results(combined_df, results_by_wave, summary)
    
    print(f"\n{'='*70}")
    print("ANÁLISE LONGITUDINAL CONCLUÍDA!")
    print("="*70)

if __name__ == "__main__":
    main()








