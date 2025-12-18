"""
Script para calcular o Alfa de Cronbach para as variáveis V307, V308 e V309
da wave 2 (sem V80).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
WAVE2_DIR = BASE_DIR / 'wave-2-1991-br'
INPUT_FILE = 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv'

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

def load_data():
    """Carrega os dados do arquivo."""
    file_path = WAVE2_DIR / INPUT_FILE
    
    if not file_path.exists():
        print(f"Arquivo não encontrado: {file_path}")
        return None
    
    print(f"Carregando dados de {file_path}...")
    
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        print(f"Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def calculate_cronbach_alpha(df, variables):
    """Calcula o Alfa de Cronbach."""
    print(f"\n{'='*60}")
    print(f"TESTE DE ALFA DE CRONBACH")
    print(f"{'='*60}\n")
    
    print(f"Variáveis: {', '.join(variables)}")
    
    # Remove linhas com valores missing
    df_clean = df[variables].dropna()
    n_valid = len(df_clean)
    
    print(f"\nCasos válidos: {n_valid} de {len(df)}")
    
    if n_valid < 10:
        print("⚠️  Poucos casos válidos para calcular alfa.")
        return None
    
    # Estatísticas descritivas
    print(f"\n{'─'*60}")
    print("ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS:")
    print(f"{'─'*60}\n")
    for var in variables:
        print(f"{var}:")
        print(f"  Média: {df_clean[var].mean():.4f}")
        print(f"  DP: {df_clean[var].std():.4f}")
        print(f"  Mín: {df_clean[var].min():.4f}")
        print(f"  Máx: {df_clean[var].max():.4f}")
        print(f"  Valores únicos: {df_clean[var].nunique()}")
    
    # Matriz de correlação
    print(f"\n{'─'*60}")
    print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
    print(f"{'─'*60}\n")
    corr_matrix = df_clean[variables].corr(method='pearson')
    print(corr_matrix.round(4))
    
    # Alfa de Cronbach
    print(f"\n{'─'*60}")
    print("ALFA DE CRONBACH:")
    print(f"{'─'*60}\n")
    
    alpha_cronbach = cronbach_alpha(df_clean, variables)
    
    if pd.isna(alpha_cronbach):
        print("❌ Não foi possível calcular o Alfa de Cronbach.")
    else:
        interp = interpret_alpha(alpha_cronbach)
        print(f"Alfa de Cronbach (α): {alpha_cronbach:.4f}")
        print(f"Interpretação: {interp}")
        
        # Interpretação detalhada
        print(f"\nInterpretação do Alfa de Cronbach:")
        print(f"  α < 0.5: Inadequada")
        print(f"  0.5 ≤ α < 0.7: Aceitável")
        print(f"  0.7 ≤ α < 0.9: Boa")
        print(f"  α ≥ 0.9: Excelente")
    
    # Correlações par a par
    print(f"\n{'─'*60}")
    print("CORRELAÇÕES PAR A PAR:")
    print(f"{'─'*60}\n")
    
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            corr = df_clean[var1].corr(df_clean[var2], method='pearson')
            print(f"{var1} vs {var2}: r = {corr:.4f}")
    
    # Salva resultados
    results = {
        'variaveis': ', '.join(variables),
        'cronbach_alpha': alpha_cronbach,
        'interpretacao': interpret_alpha(alpha_cronbach),
        'n_casos_validos': n_valid
    }
    
    result_df = pd.DataFrame([results])
    output_file = BASE_DIR / 'alfa_cronbach_v307_v308_v309_wave2.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Resultados salvos em: {output_file}")
    
    return results

def main():
    """Função principal."""
    print("="*60)
    print("ALFA DE CRONBACH: V307, V308, V309 - Wave 2 (1991)")
    print("="*60)
    
    # Carrega dados
    df = load_data()
    if df is None:
        return
    
    # Variáveis de interesse (sem V80)
    variables = ['V307', 'V308', 'V309']
    
    # Verifica se as variáveis existem
    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        print(f"\n⚠️  Variáveis não encontradas: {', '.join(missing_vars)}")
        return
    
    # Calcula Alfa de Cronbach
    results = calculate_cronbach_alpha(df, variables)
    
    if results:
        print(f"\n{'='*60}")
        print("Análise concluída com sucesso!")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()








