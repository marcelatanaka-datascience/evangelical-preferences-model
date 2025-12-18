"""
Script para calcular estatísticas descritivas de tendência central
das variáveis de religiosidade do World Values Survey.

Variáveis analisadas:
1. Saliencia religiosa
2. Religiosidade individual
3. Frequência a serviços religiosos
4. Pertencimento
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuração dos caminhos dos arquivos
BASE_DIR = Path(__file__).parent

# Mapeamento de variáveis por onda conforme especificado pelo usuário
VARIABLES_MAPPING = {
    'wave-2-1991-br': {
        'file': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V151',
        'frequencia': 'V147',
        'pertencimento': 'V20'
    },
    'wave-3-1997-br': {
        'file': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V182',
        'frequencia': 'V181',
        'pertencimento': 'V28'
    },
    'wave-5-2006-br': {
        'file': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V187',
        'frequencia': 'V186',
        'pertencimento': 'V24'
    },
    'wave-6-2014-br': {
        'file': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V147',
        'frequencia': 'V145',
        'pertencimento': 'V25'
    },
    'wave-7-2018-br': {
        'file': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'saliencia': 'Q6',
        'religiosidade_individual': 'Q173',
        'frequencia': 'Q171',
        'pertencimento': 'Q94'
    }
}


def load_wave_data(wave_dir, file_name, wave_config):
    """
    Carrega os dados de uma onda específica, lendo apenas as colunas necessárias.
    
    Parâmetros:
    -----------
    wave_dir : str
        Diretório da onda
    file_name : str
        Nome do arquivo CSV
    wave_config : dict
        Configuração da onda com as variáveis necessárias
    
    Retorna:
    --------
    DataFrame
        DataFrame com apenas as colunas especificadas
    """
    file_path = BASE_DIR / wave_dir / file_name
    
    if not file_path.exists():
        print(f"Arquivo não encontrado: {file_path}")
        return None
    
    # Define as colunas que precisamos ler
    columns_to_read = [
        wave_config['saliencia'],
        wave_config['religiosidade_individual'],
        wave_config['frequencia'],
        wave_config['pertencimento']
    ]
    
    # Adiciona coluna de ID se disponível (V1 para waves 2-6, D_INTERVIEW para wave 7)
    if wave_dir == 'wave-7-2018-br':
        id_column = 'D_INTERVIEW'
    else:
        id_column = 'V1'
    
    if id_column not in columns_to_read:
        columns_to_read.append(id_column)
    
    try:
        # Primeiro, lê apenas o cabeçalho para verificar quais colunas existem
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Remove aspas e divide por ponto-e-vírgula
            header_columns = [col.strip('"') for col in first_line.split(';')]
        
        # Verifica quais colunas realmente existem no arquivo
        available_columns = []
        missing_columns = []
        
        for col in columns_to_read:
            if col in header_columns:
                available_columns.append(col)
            else:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"  Aviso: Colunas não encontradas: {missing_columns}")
        
        if not available_columns:
            print(f"  Erro: Nenhuma das colunas necessárias foi encontrada")
            return None
        
        # Lê o CSV especificando explicitamente as colunas a serem lidas
        df = pd.read_csv(
            file_path,
            sep=';',
            encoding='utf-8',
            usecols=available_columns,
            low_memory=False,
            quotechar='"'
        )
        
        # Remove aspas dos nomes das colunas se existirem
        df.columns = df.columns.str.strip('"')
        
        print(f"  Colunas lidas: {list(df.columns)}")
        
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return df


def calculate_descriptive_stats(series, variable_name, wave_name):
    """
    Calcula estatísticas descritivas de tendência central.
    
    Retorna um dicionário com:
    - N (tamanho da amostra)
    - N válidos (sem missing)
    - Média
    - Mediana
    - Moda
    - Desvio padrão
    - Mínimo
    - Máximo
    """
    # Remove valores negativos (códigos de missing no WVS)
    clean_series = series[series >= 0].dropna()
    
    if len(clean_series) == 0:
        return {
            'onda': wave_name,
            'variavel': variable_name,
            'n_total': len(series),
            'n_validos': 0,
            'media': np.nan,
            'mediana': np.nan,
            'moda': np.nan,
            'desvio_padrao': np.nan,
            'minimo': np.nan,
            'maximo': np.nan
        }
    
    # Calcula moda
    mode_result = stats.mode(clean_series, keepdims=True)
    moda = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
    
    return {
        'onda': wave_name,
        'variavel': variable_name,
        'n_total': len(series),
        'n_validos': len(clean_series),
        'media': clean_series.mean(),
        'mediana': clean_series.median(),
        'moda': moda,
        'desvio_padrao': clean_series.std(),
        'minimo': clean_series.min(),
        'maximo': clean_series.max()
    }


def process_wave(wave_name, wave_config):
    """Processa uma onda específica e retorna estatísticas descritivas."""
    print(f"\nProcessando {wave_name}...")
    
    df = load_wave_data(wave_name, wave_config['file'], wave_config)
    if df is None:
        return []
    
    results = []
    
    # 1. Saliencia religiosa
    var_name = 'saliencia_religiosa'
    var_code = wave_config['saliencia']
    if var_code in df.columns:
        stats_dict = calculate_descriptive_stats(df[var_code], var_name, wave_name)
        results.append(stats_dict)
        print(f"  ✓ {var_name} ({var_code}): {stats_dict['n_validos']} casos válidos")
    else:
        print(f"  ✗ Variável {var_code} não encontrada para {var_name}")
    
    # 2. Religiosidade individual
    var_name = 'religiosidade_individual'
    var_code = wave_config['religiosidade_individual']
    if var_code in df.columns:
        stats_dict = calculate_descriptive_stats(df[var_code], var_name, wave_name)
        results.append(stats_dict)
        print(f"  ✓ {var_name} ({var_code}): {stats_dict['n_validos']} casos válidos")
    else:
        print(f"  ✗ Variável {var_code} não encontrada para {var_name}")
    
    # 3. Frequência a serviços religiosos
    var_name = 'frequencia_servicos_religiosos'
    var_code = wave_config['frequencia']
    if var_code in df.columns:
        stats_dict = calculate_descriptive_stats(df[var_code], var_name, wave_name)
        results.append(stats_dict)
        print(f"  ✓ {var_name} ({var_code}): {stats_dict['n_validos']} casos válidos")
    else:
        print(f"  ✗ Variável {var_code} não encontrada para {var_name}")
    
    # 4. Pertencimento
    var_name = 'pertencimento'
    var_code = wave_config['pertencimento']
    if var_code in df.columns:
        stats_dict = calculate_descriptive_stats(df[var_code], var_name, wave_name)
        results.append(stats_dict)
        print(f"  ✓ {var_name} ({var_code}): {stats_dict['n_validos']} casos válidos")
    else:
        print(f"  ✗ Variável {var_code} não encontrada para {var_name}")
    
    return results


def main():
    """Função principal que processa todas as ondas e gera estatísticas."""
    print("=" * 80)
    print("ESTATÍSTICAS DESCRITIVAS DE TENDÊNCIA CENTRAL")
    print("World Values Survey - Brasil")
    print("=" * 80)
    
    all_results = []
    
    # Processa cada onda
    for wave_name, wave_config in VARIABLES_MAPPING.items():
        results = process_wave(wave_name, wave_config)
        all_results.extend(results)
    
    if not all_results:
        print("\nErro: Nenhuma onda foi processada com sucesso.")
        return
    
    # Cria DataFrame com todos os resultados
    stats_df = pd.DataFrame(all_results)
    
    # Reordena as colunas
    stats_df = stats_df[[
        'onda', 'variavel', 'n_total', 'n_validos', 
        'media', 'mediana', 'moda', 'desvio_padrao', 'minimo', 'maximo'
    ]]
    
    # Arredonda valores numéricos
    numeric_cols = ['media', 'mediana', 'moda', 'desvio_padrao', 'minimo', 'maximo']
    stats_df[numeric_cols] = stats_df[numeric_cols].round(3)
    
    # Exibe resultados
    print("\n" + "=" * 80)
    print("RESULTADOS - ESTATÍSTICAS DESCRITIVAS")
    print("=" * 80)
    
    # Agrupa por variável para melhor visualização
    variaveis = ['saliencia_religiosa', 'religiosidade_individual', 
                 'frequencia_servicos_religiosos', 'pertencimento']
    
    for var in variaveis:
        var_data = stats_df[stats_df['variavel'] == var]
        if len(var_data) > 0:
            print(f"\n{var.upper().replace('_', ' ')}")
            print("-" * 80)
            print(var_data[['onda', 'n_validos', 'media', 'mediana', 'moda', 
                           'desvio_padrao', 'minimo', 'maximo']].to_string(index=False))
    
    # Salva resultados em CSV
    output_file = BASE_DIR / 'estatisticas_descritivas_tendencia_central.csv'
    stats_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n\nResultados salvos em: {output_file}")
    
    # Cria tabela resumo por variável (média das médias por onda)
    print("\n" + "=" * 80)
    print("RESUMO POR VARIÁVEL")
    print("=" * 80)
    
    summary_data = []
    for var in variaveis:
        var_data = stats_df[stats_df['variavel'] == var]
        if len(var_data) > 0:
            summary_data.append({
                'variavel': var,
                'num_ondas': len(var_data),
                'media_geral': var_data['media'].mean(),
                'mediana_geral': var_data['mediana'].mean(),
                'total_casos_validos': var_data['n_validos'].sum()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df[['media_geral', 'mediana_geral']] = summary_df[['media_geral', 'mediana_geral']].round(3)
    print(summary_df.to_string(index=False))
    
    # Salva resumo
    summary_file = BASE_DIR / 'resumo_estatisticas_descritivas.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"\nResumo salvo em: {summary_file}")
    
    print("\n" + "=" * 80)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 80)


if __name__ == '__main__':
    main()


