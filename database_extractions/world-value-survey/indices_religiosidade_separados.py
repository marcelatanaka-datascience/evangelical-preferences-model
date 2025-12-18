"""
Script para criar dois índices separados de religiosidade:
1. Índice de Religiosidade Subjetiva (saliencia + religiosidade_individual)
2. Índice de Religiosidade Institucional (frequencia + organizacao)

Calcula medidas de confiabilidade para cada índice separadamente.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.special import ndtr

# Importa funções do script principal
import sys
sys.path.insert(0, str(Path(__file__).parent))
from indice_religiosidade import (
    polychoric_correlation,
    tetrachoric_correlation,
    ordinal_alpha
)

# Configuração dos caminhos dos arquivos
BASE_DIR = Path(__file__).parent


def cronbach_alpha(df, items):
    """
    Calcula o Alfa de Cronbach para um conjunto de itens.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame com os dados
    items : list
        Lista com os nomes das colunas (itens) a serem avaliados
    
    Retorna:
    --------
    float
        Valor do Alfa de Cronbach (0 a 1)
    """
    # Remove linhas com valores missing em qualquer item
    df_clean = df[items].dropna()
    
    if len(df_clean) < 2:
        return np.nan
    
    k = len(items)
    
    # Calcula variâncias
    item_variances = df_clean[items].var()
    total_variance = df_clean[items].sum(axis=1).var()
    
    # Fórmula do Alfa de Cronbach
    if total_variance == 0:
        return np.nan
    
    cronbach_alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    
    return cronbach_alpha


def calculate_reliability_measures(df, items, index_name):
    """
    Calcula todas as medidas de confiabilidade para um índice.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame com os dados
    items : list
        Lista com os nomes das colunas (itens) do índice
    index_name : str
        Nome do índice (para exibição)
    
    Retorna:
    --------
    dict
        Dicionário com todas as medidas calculadas
    """
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
    
    # Cronbach's Alpha
    alpha_cronbach = cronbach_alpha(df_clean, items)
    
    # Ordinal Alpha
    alpha_ord = ordinal_alpha(df_clean, items)
    
    # Correlações entre os dois itens (se houver apenas 2)
    if len(items) == 2:
        item1, item2 = items[0], items[1]
        
        # Pearson
        pearson_corr = df_clean[item1].corr(df_clean[item2], method='pearson')
        
        # Spearman
        spearman_corr = df_clean[item1].corr(df_clean[item2], method='spearman')
        
        # Kendall Tau
        kendall_corr, _ = stats.kendalltau(df_clean[item1], df_clean[item2])
        
        # Polychoric/Tetrachoric
        polychoric_corr = polychoric_correlation(df_clean[item1], df_clean[item2])
    else:
        # Se houver mais de 2 itens, calcula média das correlações
        corr_matrix_pearson = df_clean[items].corr(method='pearson')
        corr_matrix_spearman = df_clean[items].corr(method='spearman')
        
        # Média das correlações fora da diagonal
        k = len(items)
        pearson_corr = (corr_matrix_pearson.sum().sum() - k) / (k * (k - 1))
        spearman_corr = (corr_matrix_spearman.sum().sum() - k) / (k * (k - 1))
        
        # Kendall (média)
        kendall_corrs = []
        for i in range(k):
            for j in range(i + 1, k):
                tau, _ = stats.kendalltau(df_clean[items[i]], df_clean[items[j]])
                if not np.isnan(tau):
                    kendall_corrs.append(tau)
        kendall_corr = np.mean(kendall_corrs) if kendall_corrs else np.nan
        
        # Polychoric (média)
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
    print("ÍNDICES DE RELIGIOSIDADE SEPARADOS - WORLD VALUES SURVEY")
    print("=" * 70)
    print("\nÍndices:")
    print("  1. Religiosidade Subjetiva = saliencia + religiosidade_individual")
    print("  2. Religiosidade Institucional = frequencia + organizacao")
    
    # Carrega dados do índice completo
    input_file = BASE_DIR / 'indice_religiosidade_completo.csv'
    
    if not input_file.exists():
        print(f"\nErro: Arquivo não encontrado: {input_file}")
        print("Execute primeiro o script indice_religiosidade.py")
        return
    
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"\nDados carregados: {len(df)} registros")
    
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
    
    # Calcula os índices
    for index_name, config in indices_config.items():
        items = config['items']
        abbrev = config['abbrev']
        
        # Calcula o índice (média das dimensões)
        df[f'indice_{abbrev}'] = df[items].mean(axis=1)
        
        print(f"\n{'='*70}")
        print(f"ÍNDICE: {index_name.upper()}")
        print(f"{'='*70}")
        print(f"Dimensões: {', '.join(items)}")
        
        # Estatísticas descritivas
        print(f"\nEstatísticas Descritivas:")
        print(f"  Total de registros: {len(df)}")
        print(f"  Registros completos: {df[f'indice_{abbrev}'].notna().sum()}")
        print(f"  Média: {df[f'indice_{abbrev}'].mean():.3f}")
        print(f"  Desvio padrão: {df[f'indice_{abbrev}'].std():.3f}")
        print(f"  Mínimo: {df[f'indice_{abbrev}'].min():.3f}")
        print(f"  Máximo: {df[f'indice_{abbrev}'].max():.3f}")
        
        # Por onda
        print(f"\n{'='*70}")
        print(f"MEDIDAS DE CONFIABILIDADE POR ONDA")
        print(f"{'='*70}")
        
        results_by_wave = []
        
        for wave in sorted(df['wave'].unique()):
            df_wave = df[df['wave'] == wave]
            measures = calculate_reliability_measures(df_wave, items, index_name)
            
            print(f"\n{wave}:")
            print(f"  N: {measures['n']}")
            print(f"  Cronbach's Alpha: {measures['cronbach_alpha']:.4f} ({interpret_alpha(measures['cronbach_alpha'])})")
            print(f"  Ordinal Alpha: {measures['ordinal_alpha']:.4f} ({interpret_alpha(measures['ordinal_alpha'])})")
            print(f"  Correlação Pearson: {measures['pearson_corr']:.4f}")
            print(f"  Correlação Spearman: {measures['spearman_corr']:.4f}")
            print(f"  Correlação Kendall: {measures['kendall_corr']:.4f}")
            print(f"  Correlação Polychoric: {measures['polychoric_corr']:.4f}")
            
            results_by_wave.append({
                'wave': wave,
                'indice': index_name,
                'n': measures['n'],
                'cronbach_alpha': measures['cronbach_alpha'],
                'ordinal_alpha': measures['ordinal_alpha'],
                'pearson_corr': measures['pearson_corr'],
                'spearman_corr': measures['spearman_corr'],
                'kendall_corr': measures['kendall_corr'],
                'polychoric_corr': measures['polychoric_corr']
            })
        
        # Conjunto completo
        print(f"\n{'='*70}")
        print(f"MEDIDAS DE CONFIABILIDADE - CONJUNTO COMPLETO")
        print(f"{'='*70}")
        
        measures_total = calculate_reliability_measures(df, items, index_name)
        
        print(f"\nTodas as ondas:")
        print(f"  N: {measures_total['n']}")
        print(f"  Cronbach's Alpha: {measures_total['cronbach_alpha']:.4f} ({interpret_alpha(measures_total['cronbach_alpha'])})")
        print(f"  Ordinal Alpha: {measures_total['ordinal_alpha']:.4f} ({interpret_alpha(measures_total['ordinal_alpha'])})")
        print(f"  Correlação Pearson: {measures_total['pearson_corr']:.4f}")
        print(f"  Correlação Spearman: {measures_total['spearman_corr']:.4f}")
        print(f"  Correlação Kendall: {measures_total['kendall_corr']:.4f}")
        print(f"  Correlação Polychoric: {measures_total['polychoric_corr']:.4f}")
        
        results_by_wave.append({
            'wave': 'Todas as ondas',
            'indice': index_name,
            'n': measures_total['n'],
            'cronbach_alpha': measures_total['cronbach_alpha'],
            'ordinal_alpha': measures_total['ordinal_alpha'],
            'pearson_corr': measures_total['pearson_corr'],
            'spearman_corr': measures_total['spearman_corr'],
            'kendall_corr': measures_total['kendall_corr'],
            'polychoric_corr': measures_total['polychoric_corr']
        })
        
        # Salva resultados
        results_df = pd.DataFrame(results_by_wave)
        output_file = BASE_DIR / f'confiabilidade_indice_{abbrev}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResultados salvos em: {output_file}")
    
    # Salva dataset completo com os novos índices
    output_complete = BASE_DIR / 'indices_religiosidade_separados.csv'
    df.to_csv(output_complete, index=False, encoding='utf-8')
    print(f"\n{'='*70}")
    print(f"Dataset completo com índices separados salvo em: {output_complete}")
    
    # Resumo comparativo
    print(f"\n{'='*70}")
    print("RESUMO COMPARATIVO")
    print(f"{'='*70}")
    
    summary_data = []
    for index_name, config in indices_config.items():
        abbrev = config['abbrev']
        measures_total = calculate_reliability_measures(df, config['items'], index_name)
        
        summary_data.append({
            'Indice': index_name,
            'Cronbach_Alpha': measures_total['cronbach_alpha'],
            'Ordinal_Alpha': measures_total['ordinal_alpha'],
            'Pearson': measures_total['pearson_corr'],
            'Spearman': measures_total['spearman_corr'],
            'Kendall': measures_total['kendall_corr'],
            'Polychoric': measures_total['polychoric_corr'],
            'N': measures_total['n']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nMedidas de Confiabilidade - Conjunto Completo:")
    print(summary_df.round(4).to_string(index=False))
    
    summary_file = BASE_DIR / 'resumo_comparativo_indices_separados.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"\nResumo comparativo salvo em: {summary_file}")
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()










