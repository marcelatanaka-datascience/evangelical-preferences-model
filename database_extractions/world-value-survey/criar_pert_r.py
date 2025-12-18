"""
Script para criar a variável pert_r e sua padronização z-score (Pert_r_z)
a partir das variáveis de pertinência religiosa em diferentes ondas do WVS.

Variáveis por onda:
- Wave 2 (1991): V20
- Wave 3 (1997): V28
- Wave 5 (2005): V24
- Wave 6 (2014): V25
- Wave 7 (2018): Q94
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração base
BASE_DIR = Path(__file__).parent

# Configuração de cada onda
WAVES_CONFIG = {
    'wave-2-1991-br': {
        'file_original': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'variable': 'V20',
        'year': 1991
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'variable': 'V28',
        'year': 1997
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'variable': 'V24',
        'year': 2005
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'variable': 'V25',
        'year': 2014
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'variable': 'Q94',
        'year': 2018
    }
}


def load_wave_data(wave_dir, file_name):
    """Carrega os dados de uma onda do WVS."""
    file_path = BASE_DIR / wave_dir / file_name
    
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    try:
        # Lê o CSV usando ponto-e-vírgula como delimitador
        df = pd.read_csv(
            file_path,
            sep=';',
            encoding='utf-8',
            low_memory=False,
            quotechar='"'
        )
        
        # Remove aspas dos nomes das colunas se existirem
        df.columns = df.columns.str.strip('"')
        
        print(f"  ✓ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df
        
    except Exception as e:
        print(f"  ❌ Erro ao carregar: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_pert_r(df, variable, wave_name):
    """
    Processa a variável pert_r:
    1. Converte valores negativos para missing
    2. Recodifica conforme as regras específicas de cada onda
    """
    if variable not in df.columns:
        print(f"  ❌ Variável {variable} não encontrada nas colunas")
        print(f"  Colunas disponíveis (primeiras 20): {list(df.columns[:20])}")
        return False
    
    # Passo 1: Criar cópia da variável original
    original_col = df[variable].copy()
    
    # Passo 2: Converter valores negativos para missing (NaN)
    # Primeiro, converter para numérico se necessário
    original_col = pd.to_numeric(original_col, errors='coerce')
    
    # Converter valores negativos para NaN
    pert_r = original_col.copy()
    pert_r[pert_r < 0] = np.nan
    
    print(f"  ✓ Valores negativos convertidos para missing")
    print(f"    Valores únicos originais: {sorted([x for x in original_col.dropna().unique() if x >= 0])}")
    print(f"    Valores únicos após remover negativos: {sorted([x for x in pert_r.dropna().unique()])}")
    
    # Passo 3: Recodificar conforme as regras específicas de cada onda
    if wave_name == 'wave-2-1991-br':
        # Wave 2: Original 2 → pert_r: 0, Original 1 → pert_r: 2
        pert_r = pert_r.replace({2: 0, 1: 2})
        print(f"  ✓ Recodificação Wave 2 aplicada: 2→0, 1→2")
        
    elif wave_name == 'wave-3-1997-br':
        # Wave 3: Original 1 → pert_r: 2, Original 2 → pert_r: 1, Original 3 → pert_r: 0
        pert_r = pert_r.replace({1: 2, 2: 1, 3: 0})
        print(f"  ✓ Recodificação Wave 3 aplicada: 1→2, 2→1, 3→0")
        
    elif wave_name in ['wave-5-2006-br', 'wave-6-2014-br', 'wave-7-2018-br']:
        # Waves 5, 6 e 7: manter códigos originais
        print(f"  ✓ Códigos originais mantidos para {wave_name}")
    
    # Adicionar coluna pert_r ao DataFrame
    df['pert_r'] = pert_r
    
    print(f"  ✓ Coluna 'pert_r' criada")
    print(f"    Valores únicos em pert_r: {sorted([x for x in pert_r.dropna().unique()])}")
    print(f"    Missing values: {pert_r.isna().sum()} ({pert_r.isna().sum()/len(pert_r)*100:.2f}%)")
    
    return True


def standardize_pert_r(df):
    """
    Padroniza pert_r em z-score e cria a coluna Pert_r_z.
    """
    if 'pert_r' not in df.columns:
        print(f"  ❌ Coluna 'pert_r' não encontrada")
        return False
    
    pert_r = df['pert_r'].copy()
    
    # Calcular média e desvio padrão apenas dos valores não-missing
    mean = pert_r.mean()
    std = pert_r.std()
    
    if std == 0 or pd.isna(std):
        print(f"  ⚠️  Desvio padrão é zero ou NaN. Não é possível padronizar.")
        df['Pert_r_z'] = np.nan
        return False
    
    # Calcular z-score: (x - mean) / std
    df['Pert_r_z'] = (pert_r - mean) / std
    
    print(f"  ✓ Coluna 'Pert_r_z' criada")
    print(f"    Média de pert_r: {mean:.4f}")
    print(f"    Desvio padrão de pert_r: {std:.4f}")
    print(f"    Média de Pert_r_z: {df['Pert_r_z'].mean():.6f}")
    print(f"    Desvio padrão de Pert_r_z: {df['Pert_r_z'].std():.6f}")
    
    return True


def process_wave(wave_name, config):
    """Processa uma onda completa."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()}")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    file_original = config['file_original']
    variable = config['variable']
    year = config['year']
    
    # Passo 1: Carregar dados originais
    print(f"\n[PASSO 1] Carregando dados originais...")
    df = load_wave_data(wave_dir, file_original)
    
    if df is None:
        return None
    
    # Passo 2: Processar pert_r (converter negativos para missing e recodificar)
    print(f"\n[PASSO 2] Processando variável {variable}...")
    if not process_pert_r(df, variable, wave_name):
        return None
    
    # Passo 3: Padronizar em z-score
    print(f"\n[PASSO 3] Padronizando pert_r em z-score...")
    if not standardize_pert_r(df):
        return None
    
    # Passo 4: Salvar arquivo
    print(f"\n[PASSO 4] Salvando arquivo...")
    output_file = BASE_DIR / wave_dir / file_original.replace('.csv', '_pert_r.csv')
    
    try:
        df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        print(f"  ✓ Arquivo salvo: {output_file}")
        print(f"    Total de linhas: {len(df)}")
        print(f"    Total de colunas: {len(df.columns)}")
    except Exception as e:
        print(f"  ❌ Erro ao salvar: {e}")
        return None
    
    # Estatísticas finais
    print(f"\n[RESUMO FINAL]")
    print(f"  Variável original: {variable}")
    print(f"  pert_r - Valores únicos: {sorted([x for x in df['pert_r'].dropna().unique()])}")
    print(f"  pert_r - Missing: {df['pert_r'].isna().sum()} ({df['pert_r'].isna().sum()/len(df)*100:.2f}%)")
    print(f"  Pert_r_z - Média: {df['Pert_r_z'].mean():.6f}")
    print(f"  Pert_r_z - Desvio padrão: {df['Pert_r_z'].std():.6f}")
    print(f"  Pert_r_z - Min: {df['Pert_r_z'].min():.4f}")
    print(f"  Pert_r_z - Max: {df['Pert_r_z'].max():.4f}")
    
    return df


def main():
    """Função principal."""
    print("="*70)
    print("CRIAÇÃO DA VARIÁVEL pert_r E PADRONIZAÇÃO Z-SCORE")
    print("="*70)
    
    results = {}
    
    for wave_name, config in WAVES_CONFIG.items():
        df = process_wave(wave_name, config)
        if df is not None:
            results[wave_name] = df
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*70}")
    print(f"Ondas processadas com sucesso: {len(results)}/{len(WAVES_CONFIG)}")
    
    for wave_name in results.keys():
        print(f"  ✓ {wave_name}")


if __name__ == "__main__":
    main()

