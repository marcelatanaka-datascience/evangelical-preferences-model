"""
Script para recalcular o índice Y_moral usando apenas V307, V308 e V309
(sem V80) e substituir a coluna existente.
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

def recalculate_y_moral(df, variables):
    """Recalcula o índice Y_moral como média das variáveis padronizadas."""
    print(f"\n{'='*60}")
    print(f"RECÁLCULO DO ÍNDICE Y_moral")
    print(f"{'='*60}\n")
    
    print(f"Variáveis utilizadas: {', '.join(variables)}")
    
    # Seleciona apenas as variáveis padronizadas
    data = df[variables].copy()
    
    # Calcula a média das variáveis padronizadas
    df['Y_moral'] = data.mean(axis=1, skipna=False)
    
    # Estatísticas do índice recalculado
    valid_cases = df['Y_moral'].notna().sum()
    total_cases = len(df)
    
    print(f"\nEstatísticas do índice Y_moral (recalculado):")
    print(f"  Média: {df['Y_moral'].mean():.6f}")
    print(f"  Desvio padrão: {df['Y_moral'].std():.6f}")
    print(f"  Mínimo: {df['Y_moral'].min():.6f}")
    print(f"  Máximo: {df['Y_moral'].max():.6f}")
    print(f"  Valores válidos: {valid_cases} de {total_cases}")
    
    # Mostra correlação entre as variáveis e o índice
    print(f"\n{'─'*60}")
    print("CORRELAÇÕES ENTRE VARIÁVEIS E O ÍNDICE:")
    print(f"{'─'*60}\n")
    
    for var in variables:
        # Calcula correlação apenas para casos onde ambas variáveis são válidas
        valid_data = df[[var, 'Y_moral']].dropna()
        if len(valid_data) > 0:
            corr = valid_data[var].corr(valid_data['Y_moral'])
            print(f"  {var} vs Y_moral: r = {corr:.4f}")
    
    return df

def save_output(df, output_path):
    """Salva o DataFrame com o índice recalculado."""
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
    print("RECÁLCULO DO ÍNDICE Y_moral - Wave 2 (1991)")
    print("Variáveis: V307, V308, V309 (sem V80)")
    print("="*60)
    
    # Carrega dados
    df = load_data()
    if df is None:
        return
    
    # Variáveis padronizadas para o índice (sem V80_z)
    variables = ['V307_z', 'V308_z', 'V309_z']
    
    # Verifica se as variáveis existem
    if not check_variables(df, variables):
        return
    
    # Verifica se Y_moral existe
    if 'Y_moral' in df.columns:
        print(f"\n⚠️  A coluna Y_moral já existe. Será recalculada.")
        print(f"  Valor anterior (primeiras 5 linhas):")
        print(df['Y_moral'].head())
    else:
        print(f"\n⚠️  A coluna Y_moral não existe. Será criada.")
    
    # Recalcula o índice
    df_recalculated = recalculate_y_moral(df, variables)
    
    # Salva o arquivo
    output_path = WAVE2_DIR / OUTPUT_FILE
    if save_output(df_recalculated, output_path):
        print(f"\n{'='*60}")
        print("Processo concluído com sucesso!")
        print(f"{'='*60}")
        print(f"\nArquivo atualizado: {output_path}")
        print(f"Índice Y_moral recalculado usando: {', '.join(variables)}")
        print(f"\nValor novo (primeiras 5 linhas):")
        print(df_recalculated['Y_moral'].head())

if __name__ == "__main__":
    main()








