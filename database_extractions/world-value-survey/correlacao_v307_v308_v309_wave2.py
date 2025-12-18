"""
Script para calcular a correlação de Pearson entre as variáveis V307, V308 e V309
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

def check_variables(df, variables):
    """Verifica se as variáveis existem no DataFrame."""
    missing_vars = []
    
    for var in variables:
        if var not in df.columns:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Variáveis não encontradas: {', '.join(missing_vars)}")
        print(f"\nVariáveis disponíveis que começam com 'V3':")
        v3_vars = [col for col in df.columns if col.startswith('V3')]
        print(v3_vars[:30])  # Mostra as primeiras 30
        return False
    
    return True

def calculate_correlation_matrix(df, variables):
    """Calcula a matriz de correlação entre as variáveis."""
    print(f"\n{'='*60}")
    print(f"MATRIZ DE CORRELAÇÃO: {', '.join(variables)}")
    print(f"{'='*60}\n")
    
    # Seleciona apenas as variáveis de interesse
    data = df[variables].copy()
    
    # Remove valores missing
    data_clean = data.dropna(subset=variables)
    
    print(f"Total de casos no dataset: {len(df)}")
    print(f"Casos com todas as variáveis válidas: {len(data_clean)}")
    print(f"Casos removidos (missing): {len(df) - len(data_clean)}")
    
    if len(data_clean) == 0:
        print("\n❌ Não há casos válidos para calcular a correlação.")
        return None
    
    # Estatísticas descritivas
    print(f"\n{'─'*60}")
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(f"{'─'*60}")
    for var in variables:
        print(f"\n{var}:")
        print(f"  Média: {data_clean[var].mean():.4f}")
        print(f"  Desvio Padrão: {data_clean[var].std():.4f}")
        print(f"  Mínimo: {data_clean[var].min():.4f}")
        print(f"  Máximo: {data_clean[var].max():.4f}")
        print(f"  Valores únicos: {data_clean[var].nunique()}")
    
    # Calcula matriz de correlação
    correlation_matrix = data_clean[variables].corr(method='pearson')
    
    print(f"\n{'─'*60}")
    print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
    print(f"{'─'*60}\n")
    print(correlation_matrix.round(4))
    
    # Calcula correlações individuais com p-values
    print(f"\n{'─'*60}")
    print("CORRELAÇÕES PARES COM SIGNIFICÂNCIA:")
    print(f"{'─'*60}\n")
    
    results = []
    
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            corr, p_value = pearsonr(data_clean[var1], data_clean[var2])
            
            # Interpretação
            abs_corr = abs(corr)
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
            
            direction = "positiva" if corr > 0 else "negativa"
            
            if p_value < 0.001:
                significance = "altamente significativa (p < 0.001)"
            elif p_value < 0.01:
                significance = "muito significativa (p < 0.01)"
            elif p_value < 0.05:
                significance = "significativa (p < 0.05)"
            else:
                significance = "não significativa (p >= 0.05)"
            
            print(f"{var1} vs {var2}:")
            print(f"  r = {corr:.4f}, p = {p_value:.6f}")
            print(f"  {direction} {strength}, {significance}\n")
            
            results.append({
                'variavel1': var1,
                'variavel2': var2,
                'correlacao_pearson': corr,
                'p_value': p_value,
                'n_casos_validos': len(data_clean),
                'interpretacao': f"{direction} {strength}",
                'significancia': significance
            })
    
    # Salva resultados em CSV
    result_df = pd.DataFrame(results)
    output_file = BASE_DIR / 'correlacao_v307_v308_v309_wave2.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ Resultado salvo em: {output_file}")
    
    # Salva também a matriz de correlação
    matrix_file = BASE_DIR / 'matriz_correlacao_v307_v308_v309_wave2.csv'
    correlation_matrix.to_csv(matrix_file, encoding='utf-8-sig')
    print(f"✓ Matriz de correlação salva em: {matrix_file}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'results': results,
        'n_valid': len(data_clean)
    }

def main():
    """Função principal."""
    print("="*60)
    print("CORRELAÇÃO DE PEARSON: V307, V308 e V309 - Wave 2 (1991)")
    print("="*60)
    
    # Carrega dados
    df = load_wave2_data()
    if df is None:
        return
    
    # Variáveis de interesse
    variables = ['V307', 'V308', 'V309']
    
    # Verifica se as variáveis existem
    if not check_variables(df, variables):
        return
    
    # Calcula correlações
    result = calculate_correlation_matrix(df, variables)
    
    if result:
        print(f"\n{'='*60}")
        print("Análise concluída com sucesso!")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()








