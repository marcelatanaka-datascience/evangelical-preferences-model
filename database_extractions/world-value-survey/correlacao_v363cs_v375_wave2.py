"""
Script para calcular a correlação de Pearson entre as variáveis V363 e V375
para a wave 2 do World Values Survey (Brasil 1991).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
WAVE2_DIR = BASE_DIR / 'wave-2-1991-br'
WAVE2_FILE = 'WV2_Data_Brazil_Csv_v1.6.1.csv'

def load_wave2_data():
    """Carrega os dados da wave 2."""
    file_path = WAVE2_DIR / WAVE2_FILE
    
    if not file_path.exists():
        print(f"Arquivo não encontrado: {file_path}")
        return None
    
    print(f"Carregando dados de {file_path}...")
    
    try:
        # Lê o arquivo CSV (separado por ponto-e-vírgula)
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        print(f"Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def check_variables(df, var1, var2):
    """Verifica se as variáveis existem no DataFrame."""
    missing_vars = []
    
    if var1 not in df.columns:
        missing_vars.append(var1)
    if var2 not in df.columns:
        missing_vars.append(var2)
    
    if missing_vars:
        print(f"\n⚠️  Variáveis não encontradas: {', '.join(missing_vars)}")
        print(f"\nVariáveis disponíveis que começam com 'V3':")
        v3_vars = [col for col in df.columns if col.startswith('V3')]
        print(v3_vars[:20])  # Mostra as primeiras 20
        return False
    
    return True

def calculate_pearson_correlation(df, var1, var2):
    """Calcula a correlação de Pearson entre duas variáveis."""
    print(f"\n{'='*60}")
    print(f"CORRELAÇÃO DE PEARSON: {var1} vs {var2}")
    print(f"{'='*60}\n")
    
    # Seleciona apenas as variáveis de interesse
    data = df[[var1, var2]].copy()
    
    # Remove valores missing
    data_clean = data.dropna(subset=[var1, var2])
    
    print(f"Total de casos no dataset: {len(df)}")
    print(f"Casos com ambas variáveis válidas: {len(data_clean)}")
    print(f"Casos removidos (missing): {len(df) - len(data_clean)}")
    
    if len(data_clean) == 0:
        print("\n❌ Não há casos válidos para calcular a correlação.")
        return None
    
    # Estatísticas descritivas
    print(f"\n{'─'*60}")
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(f"{'─'*60}")
    print(f"\n{var1}:")
    print(f"  Média: {data_clean[var1].mean():.4f}")
    print(f"  Desvio Padrão: {data_clean[var1].std():.4f}")
    print(f"  Mínimo: {data_clean[var1].min():.4f}")
    print(f"  Máximo: {data_clean[var1].max():.4f}")
    print(f"  Valores únicos: {data_clean[var1].nunique()}")
    
    print(f"\n{var2}:")
    print(f"  Média: {data_clean[var2].mean():.4f}")
    print(f"  Desvio Padrão: {data_clean[var2].std():.4f}")
    print(f"  Mínimo: {data_clean[var2].min():.4f}")
    print(f"  Máximo: {data_clean[var2].max():.4f}")
    print(f"  Valores únicos: {data_clean[var2].nunique()}")
    
    # Calcula correlação de Pearson
    correlation, p_value = pearsonr(data_clean[var1], data_clean[var2])
    
    print(f"\n{'─'*60}")
    print("RESULTADO DA CORRELAÇÃO:")
    print(f"{'─'*60}")
    print(f"\nCoeficiente de Correlação de Pearson (r): {correlation:.4f}")
    print(f"Valor-p (p-value): {p_value:.6f}")
    
    # Interpretação
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        strength = "muito fraca"
    elif abs_corr < 0.3:
        strength = "fraca"
    elif abs_corr < 0.5:
        strength = "moderada"
    elif abs_corr < 0.7:
        strength = "forte"
    else:
        strength = "muito forte"
    
    direction = "positiva" if correlation > 0 else "negativa"
    
    print(f"\nInterpretação:")
    print(f"  Correlação {direction} {strength}")
    
    if p_value < 0.001:
        significance = "altamente significativa (p < 0.001)"
    elif p_value < 0.01:
        significance = "muito significativa (p < 0.01)"
    elif p_value < 0.05:
        significance = "significativa (p < 0.05)"
    else:
        significance = "não significativa (p >= 0.05)"
    
    print(f"  {significance}")
    
    # Salva resultado em CSV
    result_df = pd.DataFrame({
        'variavel1': [var1],
        'variavel2': [var2],
        'correlacao_pearson': [correlation],
        'p_value': [p_value],
        'n_casos_validos': [len(data_clean)],
        'interpretacao': [f"{direction} {strength}"]
    })
    
    output_file = BASE_DIR / 'correlacao_v363_v375_wave2.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Resultado salvo em: {output_file}")
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_valid': len(data_clean),
        'var1': var1,
        'var2': var2
    }

def main():
    """Função principal."""
    print("="*60)
    print("CORRELAÇÃO DE PEARSON: V363 vs V375 - Wave 2 (1991)")
    print("="*60)
    
    # Carrega dados
    df = load_wave2_data()
    if df is None:
        return
    
    # Variáveis de interesse
    var1 = 'V363'
    var2 = 'V375'
    
    # Verifica se as variáveis existem
    if not check_variables(df, var1, var2):
        return
    
    # Calcula correlação
    result = calculate_pearson_correlation(df, var1, var2)
    
    if result:
        print(f"\n{'='*60}")
        print("Análise concluída com sucesso!")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

