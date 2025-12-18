"""
Script para criar um índice único (Y_moral) a partir das 4 variáveis padronizadas
(V307_z, V308_z, V309_z, V80_z) da wave 2.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
WAVE2_DIR = BASE_DIR / 'wave-2-1991-br'
INPUT_FILE = 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv'
OUTPUT_FILE = 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv'

def load_data():
    """Carrega os dados do arquivo com z-scores."""
    file_path = WAVE2_DIR / INPUT_FILE
    
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

def create_index(df, variables, index_name):
    """Cria um índice como média das variáveis padronizadas."""
    print(f"\n{'='*60}")
    print(f"CRIAÇÃO DO ÍNDICE: {index_name}")
    print(f"{'='*60}\n")
    
    # Verifica se o índice já existe
    if index_name in df.columns:
        print(f"⚠️  A coluna {index_name} já existe. Será recalculada.")
    
    # Seleciona apenas as variáveis de interesse
    data = df[variables].copy()
    
    # Calcula a média das variáveis padronizadas
    # Usa skipna=True para ignorar valores missing em algumas variáveis
    df[index_name] = data.mean(axis=1, skipna=False)
    
    # Estatísticas do índice
    valid_cases = df[index_name].notna().sum()
    total_cases = len(df)
    
    print(f"Variáveis utilizadas: {', '.join(variables)}")
    print(f"\nEstatísticas do índice {index_name}:")
    print(f"  Média: {df[index_name].mean():.6f}")
    print(f"  Desvio padrão: {df[index_name].std():.6f}")
    print(f"  Mínimo: {df[index_name].min():.6f}")
    print(f"  Máximo: {df[index_name].max():.6f}")
    print(f"  Valores válidos: {valid_cases} de {total_cases}")
    
    # Mostra correlação entre as variáveis e o índice
    print(f"\n{'─'*60}")
    print("CORRELAÇÕES ENTRE VARIÁVEIS E O ÍNDICE:")
    print(f"{'─'*60}\n")
    
    for var in variables:
        # Calcula correlação apenas para casos onde ambas variáveis são válidas
        valid_data = df[[var, index_name]].dropna()
        if len(valid_data) > 0:
            corr = valid_data[var].corr(valid_data[index_name])
            print(f"  {var} vs {index_name}: r = {corr:.4f}")
    
    # Estatísticas descritivas por número de variáveis válidas
    print(f"\n{'─'*60}")
    print("ESTATÍSTICAS POR NÚMERO DE VARIÁVEIS VÁLIDAS:")
    print(f"{'─'*60}\n")
    
    # Conta quantas variáveis são válidas para cada caso
    n_valid = data.notna().sum(axis=1)
    df['n_valid_vars'] = n_valid
    
    for n in range(1, len(variables) + 1):
        subset = df[df['n_valid_vars'] == n]
        if len(subset) > 0:
            print(f"  {n} variável(is) válida(s): {len(subset)} casos")
            print(f"    Média do índice: {subset[index_name].mean():.6f}")
            print(f"    DP do índice: {subset[index_name].std():.6f}")
    
    # Remove a coluna auxiliar
    df = df.drop(columns=['n_valid_vars'])
    
    return df

def save_output(df, output_path):
    """Salva o DataFrame com o novo índice."""
    try:
        # Salva mantendo o separador original (ponto-e-vírgula)
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"\n✓ Arquivo salvo com sucesso: {output_path}")
        print(f"  Total de colunas: {len(df.columns)}")
        print(f"  Total de linhas: {len(df)}")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")
        return False

def main():
    """Função principal."""
    print("="*60)
    print("CRIAÇÃO DO ÍNDICE Y_moral - Wave 2 (1991)")
    print("="*60)
    
    # Carrega dados
    df = load_data()
    if df is None:
        return
    
    # Variáveis padronizadas para o índice
    variables = ['V307_z', 'V308_z', 'V309_z', 'V80_z']
    index_name = 'Y_moral'
    
    # Verifica se as variáveis existem
    if not check_variables(df, variables):
        return
    
    # Cria o índice
    df_with_index = create_index(df, variables, index_name)
    
    # Salva o arquivo
    output_path = WAVE2_DIR / OUTPUT_FILE
    if save_output(df_with_index, output_path):
        print(f"\n{'='*60}")
        print("Processo concluído com sucesso!")
        print(f"{'='*60}")
        print(f"\nArquivo atualizado: {output_path}")
        print(f"Índice criado: {index_name}")
        print(f"\nÍndice calculado como média de: {', '.join(variables)}")

if __name__ == "__main__":
    main()








