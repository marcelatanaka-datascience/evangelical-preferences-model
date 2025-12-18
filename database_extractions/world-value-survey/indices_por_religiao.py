"""
Script para criar índices de religiosidade separados por categoria religiosa:
1. Evangélicos (incluindo todas as subdenominações)
2. Católicos
3. Outras religiões

Calcula os índices de Religiosidade Subjetiva e Institucional para cada grupo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# Importa funções do script principal
import sys
sys.path.insert(0, str(Path(__file__).parent))
from indice_religiosidade import (
    polychoric_correlation,
    ordinal_alpha
)

# Configuração dos caminhos dos arquivos
BASE_DIR = Path(__file__).parent

# Mapeamento de variáveis de religião por onda
RELIGION_VARIABLES = {
    'wave-2-1991-br': {
        'religion': 'V144',
        'denomination': None  # Wave 2 não tem variável de denominação específica
    },
    'wave-3-1997-br': {
        'religion': 'V180',
        'denomination': None
    },
    'wave-5-2006-br': {
        'religion': 'V185',
        'denomination': None
    },
    'wave-6-2014-br': {
        'religion': 'V146',
        'denomination': None
    },
    'wave-7-2018-br': {
        'religion': 'Q289',
        'denomination': 'Q290'  # Denominação específica
    }
}

# Mapeamento de variáveis por onda (do script principal)
VARIABLES_MAPPING = {
    'wave-2-1991-br': {
        'file': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V151',
        'frequencia': 'V147',
        'organizacao': 'V20'
    },
    'wave-3-1997-br': {
        'file': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V182',
        'frequencia': 'V181',
        'organizacao': 'V28'
    },
    'wave-5-2006-br': {
        'file': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V187',
        'frequencia': 'V186',
        'organizacao': 'V24'
    },
    'wave-6-2014-br': {
        'file': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V147',
        'frequencia': 'V145',
        'organizacao': 'V25'
    },
    'wave-7-2018-br': {
        'file': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'saliencia': 'Q6',
        'religiosidade_individual': 'Q173',
        'frequencia': 'Q171',
        'organizacao': 'Q94'
    }
}


def load_wave_data(wave_dir, file_name, wave_config=None, religion_config=None):
    """
    Carrega os dados de uma onda específica, lendo apenas as colunas necessárias.
    
    Parâmetros:
    -----------
    wave_dir : str
        Diretório da onda
    file_name : str
        Nome do arquivo CSV
    wave_config : dict, opcional
        Configuração da onda com as variáveis de religiosidade
    religion_config : dict, opcional
        Configuração da onda com as variáveis de religião
    
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
    columns_to_read = []
    
    if wave_config:
        columns_to_read.extend([
            wave_config['saliencia'],
            wave_config['religiosidade_individual'],
            wave_config['frequencia'],
            wave_config['organizacao']
        ])
    
    if religion_config:
        columns_to_read.append(religion_config['religion'])
        if religion_config.get('denomination'):
            columns_to_read.append(religion_config['denomination'])
    
    # Adiciona coluna de ID se disponível
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


def map_religion_category(value, wave_name, denomination_value=None):
    """
    Mapeia valores de religião para categorias: Evangélicos, Católicos, Outras.
    
    No WVS:
    - 1 = Católico Romano
    - 2 = Protestante/Evangélico
    - Outros valores = Outras religiões
    
    Para Wave 2 e 7, os códigos são diferentes:
    - 10100000 ou similar = Católico
    - 20000000 ou similar = Protestante/Evangélico
    """
    if pd.isna(value) or value < 0:
        return 'Outras'
    
    try:
        val = float(value)
        
        # Waves 3, 5, 6: códigos simples
        if wave_name in ['wave-3-1997-br', 'wave-5-2006-br', 'wave-6-2014-br']:
            if val == 1.0:
                return 'Católicos'
            elif val == 2.0:
                # Verifica se há denominação específica para identificar subdenominações evangélicas
                if denomination_value is not None and not pd.isna(denomination_value):
                    # Todas as subdenominações protestantes são consideradas evangélicas
                    return 'Evangélicos'
                return 'Evangélicos'
            else:
                return 'Outras'
        
        # Wave 2: códigos complexos
        elif wave_name == 'wave-2-1991-br':
            # 10100000 = Católico Romano
            if val == 10100000.0:
                return 'Católicos'
            # 20000000, 20213020, etc. = Protestante/Evangélico
            elif val >= 20000000.0 and val < 30000000.0:
                return 'Evangélicos'
            else:
                return 'Outras'
        
        # Wave 7: códigos complexos
        elif wave_name == 'wave-7-2018-br':
            # 10100000 = Católico Romano
            if val == 10100000.0:
                return 'Católicos'
            # 20000000, 20213020, 21400000, etc. = Protestante/Evangélico
            # Inclui: Batistas, Luteranos, Metodistas, Pentecostais, Assembleia de Deus, etc.
            elif val >= 20000000.0 and val < 30000000.0:
                # Verifica denominação específica (Q290) para confirmar
                if denomination_value is not None:
                    # Q290: 1=Católico, 2=Protestante Histórico, 3=Pentecostal, 4=Neopentecostal, etc.
                    # Todos são evangélicos
                    return 'Evangélicos'
                return 'Evangélicos'
            else:
                return 'Outras'
        
        return 'Outras'
    except:
        return 'Outras'


def process_wave_with_religion(wave_name, wave_config, religion_config):
    """Processa uma onda e adiciona categoria religiosa."""
    wave_dir = wave_name
    file_name = wave_config['file']
    
    df = load_wave_data(wave_dir, file_name, wave_config, religion_config)
    if df is None:
        return None
    
    # Adiciona identificador de onda
    df['wave'] = wave_name
    
    # Mapeia religião
    religion_var = religion_config['religion']
    denomination_var = religion_config.get('denomination')
    
    if religion_var not in df.columns:
        print(f"  Variável de religião {religion_var} não encontrada em {wave_name}")
        return None
    
    # Mapeia categoria religiosa
    if denomination_var and denomination_var in df.columns:
        df['religiao_categoria'] = df.apply(
            lambda row: map_religion_category(
                row[religion_var], 
                wave_name, 
                row[denomination_var] if not pd.isna(row[denomination_var]) else None
            ),
            axis=1
        )
    else:
        df['religiao_categoria'] = df[religion_var].apply(
            lambda x: map_religion_category(x, wave_name)
        )
    
    # Carrega dados normalizados do índice completo
    indices_file = BASE_DIR / 'indices_religiosidade_separados.csv'
    if indices_file.exists():
        indices_df = pd.read_csv(indices_file, encoding='utf-8')
        # Faz merge baseado em wave e id_original se disponível
        # Para simplificar, vamos recalcular as dimensões normalizadas
        pass
    
    return df


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
    
    cronbach_alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return cronbach_alpha


def calculate_reliability_measures(df, items, index_name):
    """Calcula todas as medidas de confiabilidade para um índice."""
    df_clean = df[items].dropna()
    
    if len(df_clean) < 10:
        return {
            'cronbach_alpha': np.nan,
            'ordinal_alpha': np.nan,
            'pearson_corr': np.nan,
            'spearman_corr': np.nan,
            'kendall_corr': np.nan,
            'polychoric_corr': np.nan,
            'n': len(df_clean)
        }
    
    alpha_cronbach = cronbach_alpha(df_clean, items)
    alpha_ord = ordinal_alpha(df_clean, items)
    
    if len(items) == 2:
        item1, item2 = items[0], items[1]
        pearson_corr = df_clean[item1].corr(df_clean[item2], method='pearson')
        spearman_corr = df_clean[item1].corr(df_clean[item2], method='spearman')
        kendall_corr, _ = stats.kendalltau(df_clean[item1], df_clean[item2])
        polychoric_corr = polychoric_correlation(df_clean[item1], df_clean[item2])
    else:
        corr_matrix_pearson = df_clean[items].corr(method='pearson')
        corr_matrix_spearman = df_clean[items].corr(method='spearman')
        k = len(items)
        pearson_corr = (corr_matrix_pearson.sum().sum() - k) / (k * (k - 1))
        spearman_corr = (corr_matrix_spearman.sum().sum() - k) / (k * (k - 1))
        
        kendall_corrs = []
        for i in range(k):
            for j in range(i + 1, k):
                tau, _ = stats.kendalltau(df_clean[items[i]], df_clean[items[j]])
                if not np.isnan(tau):
                    kendall_corrs.append(tau)
        kendall_corr = np.mean(kendall_corrs) if kendall_corrs else np.nan
        
        polychoric_corrs = []
        for i in range(k):
            for j in range(i + 1, k):
                rho = polychoric_correlation(df_clean[items[i]], df_clean[items[j]])
                if not np.isnan(rho):
                    polychoric_corrs.append(rho)
        polychoric_corr = np.mean(polychoric_corrs) if polychoric_corrs else np.nan
    
    return {
        'cronbach_alpha': alpha_cronbach,
        'ordinal_alpha': alpha_ord,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'kendall_corr': kendall_corr,
        'polychoric_corr': polychoric_corr,
        'n': len(df_clean)
    }


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


def main():
    """Função principal."""
    print("=" * 70)
    print("ÍNDICES DE RELIGIOSIDADE POR CATEGORIA RELIGIOSA")
    print("=" * 70)
    print("\nCategorias:")
    print("  1. Evangélicos (incluindo todas as subdenominações)")
    print("  2. Católicos")
    print("  3. Outras religiões")
    
    # Carrega dados do índice completo (já normalizados)
    indices_file = BASE_DIR / 'indices_religiosidade_separados.csv'
    
    if not indices_file.exists():
        print(f"\nErro: Arquivo não encontrado: {indices_file}")
        print("Execute primeiro o script indices_religiosidade_separados.py")
        return
    
    df_indices = pd.read_csv(indices_file, encoding='utf-8')
    print(f"\nDados carregados: {len(df_indices)} registros")
    
    # Processa cada onda para adicionar categoria religiosa
    print("\nProcessando ondas para identificar categoria religiosa...")
    
    all_results = []
    
    for wave_name in VARIABLES_MAPPING.keys():
        wave_config = VARIABLES_MAPPING[wave_name]
        religion_config = RELIGION_VARIABLES[wave_name]
        
        print(f"Processando {wave_name}...")
        df_wave = process_wave_with_religion(wave_name, wave_config, religion_config)
        
        if df_wave is None:
            continue
        
        # Carrega dados normalizados para esta onda
        df_indices_wave = df_indices[df_indices['wave'] == wave_name].copy()
        
        # Adiciona categoria religiosa baseado na ordem (assumindo mesma ordem dos dados)
        # Os dados normalizados devem estar na mesma ordem que os dados originais
        if len(df_wave) == len(df_indices_wave):
            df_indices_wave['religiao_categoria'] = df_wave['religiao_categoria'].values
            all_results.append(df_indices_wave)
        else:
            print(f"  Aviso: Tamanhos diferentes ({len(df_wave)} vs {len(df_indices_wave)})")
            # Se os tamanhos são diferentes, tenta fazer merge por índice
            # Assume que os primeiros N registros correspondem
            min_len = min(len(df_wave), len(df_indices_wave))
            df_indices_wave = df_indices_wave.iloc[:min_len].copy()
            df_indices_wave['religiao_categoria'] = df_wave['religiao_categoria'].iloc[:min_len].values
            all_results.append(df_indices_wave)
    
    if not all_results:
        print("\nErro: Nenhuma onda foi processada com sucesso.")
        return
    
    # Combina todos os resultados
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print(f"\nTotal de registros com categoria religiosa: {len(combined_df)}")
    print("\nDistribuição por categoria religiosa:")
    print(combined_df['religiao_categoria'].value_counts())
    print("\nPercentual por categoria:")
    print(combined_df['religiao_categoria'].value_counts(normalize=True) * 100)
    
    # Define os dois índices
    indices_config = {
        'Religiosidade Subjetiva': {
            'items': ['saliencia', 'religiosidade_individual'],
            'abbrev': 'subjetiva'
        },
        'Religiosidade Institucional': {
            'items': ['frequencia', 'organizacao'],
            'abbrev': 'institucional'
        }
    }
    
    # Calcula índices e medidas de confiabilidade por categoria religiosa
    print("\n" + "=" * 70)
    print("ANÁLISE POR CATEGORIA RELIGIOSA")
    print("=" * 70)
    
    all_reliability_results = []
    
    for categoria in ['Evangélicos', 'Católicos', 'Outras']:
        df_categoria = combined_df[combined_df['religiao_categoria'] == categoria]
        
        if len(df_categoria) == 0:
            print(f"\n{categoria}: Nenhum registro encontrado")
            continue
        
        print(f"\n{'='*70}")
        print(f"CATEGORIA: {categoria.upper()}")
        print(f"{'='*70}")
        print(f"Total de registros: {len(df_categoria)}")
        
        for index_name, config in indices_config.items():
            items = config['items']
            abbrev = config['abbrev']
            
            print(f"\n{index_name}:")
            print("-" * 70)
            
            # Por onda
            results_by_wave = []
            
            for wave in sorted(df_categoria['wave'].unique()):
                df_wave_cat = df_categoria[df_categoria['wave'] == wave]
                measures = calculate_reliability_measures(df_wave_cat, items, index_name)
                
                if measures['n'] < 10:
                    continue
                
                print(f"  {wave}:")
                print(f"    N: {measures['n']}")
                print(f"    Cronbach's Alpha: {measures['cronbach_alpha']:.4f} ({interpret_alpha(measures['cronbach_alpha'])})")
                print(f"    Ordinal Alpha: {measures['ordinal_alpha']:.4f} ({interpret_alpha(measures['ordinal_alpha'])})")
                print(f"    Correlação Pearson: {measures['pearson_corr']:.4f}")
                print(f"    Correlação Spearman: {measures['spearman_corr']:.4f}")
                print(f"    Correlação Kendall: {measures['kendall_corr']:.4f}")
                print(f"    Correlação Polychoric: {measures['polychoric_corr']:.4f}")
                
                results_by_wave.append({
                    'categoria': categoria,
                    'indice': index_name,
                    'wave': wave,
                    'n': measures['n'],
                    'cronbach_alpha': measures['cronbach_alpha'],
                    'ordinal_alpha': measures['ordinal_alpha'],
                    'pearson_corr': measures['pearson_corr'],
                    'spearman_corr': measures['spearman_corr'],
                    'kendall_corr': measures['kendall_corr'],
                    'polychoric_corr': measures['polychoric_corr']
                })
            
            # Conjunto completo
            measures_total = calculate_reliability_measures(df_categoria, items, index_name)
            
            if measures_total['n'] >= 10:
                print(f"\n  Todas as ondas ({categoria}):")
                print(f"    N: {measures_total['n']}")
                print(f"    Cronbach's Alpha: {measures_total['cronbach_alpha']:.4f} ({interpret_alpha(measures_total['cronbach_alpha'])})")
                print(f"    Ordinal Alpha: {measures_total['ordinal_alpha']:.4f} ({interpret_alpha(measures_total['ordinal_alpha'])})")
                print(f"    Correlação Pearson: {measures_total['pearson_corr']:.4f}")
                print(f"    Correlação Spearman: {measures_total['spearman_corr']:.4f}")
                print(f"    Correlação Kendall: {measures_total['kendall_corr']:.4f}")
                print(f"    Correlação Polychoric: {measures_total['polychoric_corr']:.4f}")
                
                results_by_wave.append({
                    'categoria': categoria,
                    'indice': index_name,
                    'wave': 'Todas as ondas',
                    'n': measures_total['n'],
                    'cronbach_alpha': measures_total['cronbach_alpha'],
                    'ordinal_alpha': measures_total['ordinal_alpha'],
                    'pearson_corr': measures_total['pearson_corr'],
                    'spearman_corr': measures_total['spearman_corr'],
                    'kendall_corr': measures_total['kendall_corr'],
                    'polychoric_corr': measures_total['polychoric_corr']
                })
            
            all_reliability_results.extend(results_by_wave)
    
    # Salva resultados
    if all_reliability_results:
        results_df = pd.DataFrame(all_reliability_results)
        output_file = BASE_DIR / 'confiabilidade_indices_por_religiao.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n{'='*70}")
        print(f"Resultados salvos em: {output_file}")
    
    # Salva dataset completo com categoria religiosa
    output_complete = BASE_DIR / 'indices_religiosidade_por_religiao.csv'
    combined_df.to_csv(output_complete, index=False, encoding='utf-8')
    print(f"Dataset completo com categoria religiosa salvo em: {output_complete}")
    
    # Resumo comparativo
    print(f"\n{'='*70}")
    print("RESUMO COMPARATIVO POR CATEGORIA RELIGIOSA")
    print(f"{'='*70}")
    
    summary_data = []
    for categoria in ['Evangélicos', 'Católicos', 'Outras']:
        df_cat = combined_df[combined_df['religiao_categoria'] == categoria]
        if len(df_cat) == 0:
            continue
        
        for index_name, config in indices_config.items():
            items = config['items']
            measures = calculate_reliability_measures(df_cat, items, index_name)
            
            if measures['n'] >= 10:
                summary_data.append({
                    'Categoria': categoria,
                    'Indice': index_name,
                    'Cronbach_Alpha': measures['cronbach_alpha'],
                    'Ordinal_Alpha': measures['ordinal_alpha'],
                    'Pearson': measures['pearson_corr'],
                    'Spearman': measures['spearman_corr'],
                    'Kendall': measures['kendall_corr'],
                    'Polychoric': measures['polychoric_corr'],
                    'N': measures['n']
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\nMedidas de Confiabilidade - Conjunto Completo por Categoria:")
        print(summary_df.round(4).to_string(index=False))
        
        summary_file = BASE_DIR / 'resumo_comparativo_por_religiao.csv'
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"\nResumo comparativo salvo em: {summary_file}")
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

