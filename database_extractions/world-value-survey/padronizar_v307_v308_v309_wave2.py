"""
Script para padronizar as variáveis V307, V308 e V309 em z-score
e adicionar as colunas padronizadas ao CSV original da wave 2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
WAVE2_DIR = BASE_DIR / 'wave-2-1991-br'
WAVE2_FILE = 'WV2_Data_Brazil_Csv_v1.6.1.csv'
OUTPUT_FILE = 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv'

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
        return False
    
    return True

def standardize_variables(df, variables):
    """Padroniza as variáveis em z-score e adiciona colunas ao DataFrame."""
    print(f"\n{'='*60}")
    print(f"PADRONIZAÇÃO EM Z-SCORE: {', '.join(variables)}")
    print(f"{'='*60}\n")
    
    # Cria uma cópia do DataFrame
    df_copy = df.copy()
    
    # Padroniza cada variável
    for var in variables:
        z_col = f"{var}_z"
        
        # Calcula z-score (subtrai média e divide pelo desvio padrão)
        # Usa apenas valores válidos (não missing) para calcular média e desvio
        valid_values = df_copy[var].dropna()
        
        if len(valid_values) == 0:
            print(f"⚠️  Variável {var} não tem valores válidos. Pulando...")
            continue
        
        mean = valid_values.mean()
        std = valid_values.std()
        
        # Calcula z-score: (x - mean) / std
        # Valores missing permanecem como missing
        df_copy[z_col] = (df_copy[var] - mean) / std
        
        print(f"{var} -> {z_col}:")
        print(f"  Média original: {mean:.4f}")
        print(f"  Desvio padrão original: {std:.4f}")
        print(f"  Média z-score: {df_copy[z_col].mean():.6f}")
        print(f"  Desvio padrão z-score: {df_copy[z_col].std():.6f}")
        print(f"  Valores válidos: {df_copy[z_col].notna().sum()}")
        print()
    
    return df_copy

def save_output(df, output_path):
    """Salva o DataFrame com as novas colunas."""
    try:
        # Salva mantendo o separador original (ponto-e-vírgula)
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"✓ Arquivo salvo com sucesso: {output_path}")
        print(f"  Total de colunas: {len(df.columns)}")
        print(f"  Total de linhas: {len(df)}")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")
        return False

def main():
    """Função principal."""
    print("="*60)
    print("PADRONIZAÇÃO EM Z-SCORE: V307, V308 e V309 - Wave 2 (1991)")
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
    
    # Padroniza as variáveis
    df_with_zscore = standardize_variables(df, variables)
    
    # Salva o arquivo
    output_path = WAVE2_DIR / OUTPUT_FILE
    if save_output(df_with_zscore, output_path):
        print(f"\n{'='*60}")
        print("Processo concluído com sucesso!")
        print(f"{'='*60}")
        print(f"\nArquivo original: {WAVE2_DIR / WAVE2_FILE}")
        print(f"Arquivo com z-scores: {output_path}")
        print(f"\nColunas adicionadas:")
        for var in variables:
            print(f"  - {var}_z")

if __name__ == "__main__":
    main()








