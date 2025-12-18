"""
Script para calcular correlação entre Pert_r_z e Freq_inv_z
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Configuração dos arquivos
WAVES_CONFIG = {
    'wave-2-1991-br': {
        'pert_r_file': 'WV2_Data_Brazil_Csv_v1.6.1_pert_r.csv',
        'freq_inv_file': 'WV2_Data_Brazil_Csv_v1.6.1_com_freq_inv.csv',
        'year': 1991
    },
    'wave-3-1997-br': {
        'pert_r_file': 'WV3_Data_Brazil_Csv_v20221107.1_pert_r.csv',
        'freq_inv_file': 'WV3_Data_Brazil_Csv_v20221107.1_com_freq_inv.csv',
        'year': 1997
    },
    'wave-5-2006-br': {
        'pert_r_file': 'WV5_Data_Brazil_Csv_v20201117.1_pert_r.csv',
        'freq_inv_file': 'WV5_Data_Brazil_Csv_v20201117.1_com_freq_inv.csv',
        'year': 2005
    },
    'wave-6-2014-br': {
        'pert_r_file': 'WV6_Data_Brazil_Csv_v20221117.1_pert_r.csv',
        'freq_inv_file': 'WV6_Data_Brazil_Csv_v20221117.1_com_freq_inv.csv',
        'year': 2014
    },
    'wave-7-2018-br': {
        'pert_r_file': 'WVS_Wave_7_Brazil_Csv_v5.1_pert_r.csv',
        'freq_inv_file': 'WVS_Wave_7_Brazil_Csv_v5.1_com_freq_inv.csv',
        'year': 2018
    }
}


def calculate_correlation(df, var1, var2):
    """Calcula correlações entre duas variáveis."""
    # Remover missing values
    data = df[[var1, var2]].dropna()
    
    if len(data) < 3:
        return None
    
    x = data[var1]
    y = data[var2]
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Kendall correlation
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    return {
        'n': len(data),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }


def process_wave(wave_name, config):
    """Processa uma onda e calcula correlações."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()} ({config['year']})")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    
    # Carregar arquivo com pert_r
    pert_r_path = BASE_DIR / wave_dir / config['pert_r_file']
    if not pert_r_path.exists():
        print(f"  ⚠️  Arquivo não encontrado: {pert_r_path}")
        return None
    
    # Carregar arquivo com freq_inv
    freq_inv_path = BASE_DIR / wave_dir / config['freq_inv_file']
    if not freq_inv_path.exists():
        print(f"  ⚠️  Arquivo não encontrado: {freq_inv_path}")
        return None
    
    try:
        # Carregar dados
        df_pert = pd.read_csv(pert_r_path, sep=';', encoding='utf-8', low_memory=False)
        df_freq = pd.read_csv(freq_inv_path, sep=';', encoding='utf-8', low_memory=False)
        
        print(f"  ✓ Arquivos carregados")
        print(f"    pert_r: {len(df_pert)} linhas")
        print(f"    freq_inv: {len(df_freq)} linhas")
        
        # Verificar se as colunas existem
        if 'Pert_r_z' not in df_pert.columns:
            print(f"  ❌ Coluna 'Pert_r_z' não encontrada")
            return None
        
        if 'Freq_inv_z' not in df_freq.columns:
            print(f"  ❌ Coluna 'Freq_inv_z' não encontrada")
            return None
        
        # Combinar dados - assumindo que as linhas estão na mesma ordem
        # Ou podemos tentar fazer merge por alguma coluna de ID se existir
        # Por enquanto, vamos assumir mesma ordem
        if len(df_pert) != len(df_freq):
            print(f"  ⚠️  Número de linhas diferente! Tentando merge...")
            # Tentar encontrar coluna de ID comum
            common_cols = set(df_pert.columns) & set(df_freq.columns)
            id_cols = [c for c in common_cols if c in ['V1', 'V2', 'S001', 'S002'] or 'ID' in c.upper()]
            
            if id_cols:
                print(f"    Usando coluna(s) de ID: {id_cols}")
                df = pd.merge(df_pert[['Pert_r_z'] + id_cols], 
                            df_freq[['Freq_inv_z'] + id_cols],
                            on=id_cols, how='inner')
            else:
                print(f"    Sem coluna de ID comum. Usando índice.")
                df = pd.DataFrame({
                    'Pert_r_z': df_pert['Pert_r_z'].values,
                    'Freq_inv_z': df_freq['Freq_inv_z'].values
                })
        else:
            df = pd.DataFrame({
                'Pert_r_z': df_pert['Pert_r_z'].values,
                'Freq_inv_z': df_freq['Freq_inv_z'].values
            })
        
        print(f"  ✓ Dados combinados: {len(df)} linhas")
        
        # Estatísticas descritivas
        print(f"\n[ESTATÍSTICAS DESCRITIVAS]")
        print(f"  Pert_r_z:")
        print(f"    Média: {df['Pert_r_z'].mean():.4f}")
        print(f"    Desvio padrão: {df['Pert_r_z'].std():.4f}")
        print(f"    Min: {df['Pert_r_z'].min():.4f}, Max: {df['Pert_r_z'].max():.4f}")
        print(f"    Missing: {df['Pert_r_z'].isna().sum()} ({df['Pert_r_z'].isna().sum()/len(df)*100:.2f}%)")
        
        print(f"  Freq_inv_z:")
        print(f"    Média: {df['Freq_inv_z'].mean():.4f}")
        print(f"    Desvio padrão: {df['Freq_inv_z'].std():.4f}")
        print(f"    Min: {df['Freq_inv_z'].min():.4f}, Max: {df['Freq_inv_z'].max():.4f}")
        print(f"    Missing: {df['Freq_inv_z'].isna().sum()} ({df['Freq_inv_z'].isna().sum()/len(df)*100:.2f}%)")
        
        # Calcular correlações
        print(f"\n[CORRELAÇÕES]")
        corr_results = calculate_correlation(df, 'Pert_r_z', 'Freq_inv_z')
        
        if corr_results:
            print(f"  N de casos válidos: {corr_results['n']}")
            print(f"\n  Pearson (r):")
            print(f"    r = {corr_results['pearson_r']:.4f}")
            print(f"    p = {corr_results['pearson_p']:.6f}")
            print(f"    {'***' if corr_results['pearson_p'] < 0.001 else '**' if corr_results['pearson_p'] < 0.01 else '*' if corr_results['pearson_p'] < 0.05 else 'ns'}")
            
            print(f"\n  Spearman (ρ):")
            print(f"    ρ = {corr_results['spearman_r']:.4f}")
            print(f"    p = {corr_results['spearman_p']:.6f}")
            print(f"    {'***' if corr_results['spearman_p'] < 0.001 else '**' if corr_results['spearman_p'] < 0.01 else '*' if corr_results['spearman_p'] < 0.05 else 'ns'}")
            
            print(f"\n  Kendall (τ):")
            print(f"    τ = {corr_results['kendall_tau']:.4f}")
            print(f"    p = {corr_results['kendall_p']:.6f}")
            print(f"    {'***' if corr_results['kendall_p'] < 0.001 else '**' if corr_results['kendall_p'] < 0.01 else '*' if corr_results['kendall_p'] < 0.05 else 'ns'}")
            
            return {
                'wave': wave_name,
                'year': config['year'],
                **corr_results
            }
        else:
            print(f"  ⚠️  Não foi possível calcular correlações (poucos dados válidos)")
            return None
            
    except Exception as e:
        print(f"  ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Função principal."""
    print("="*70)
    print("CORRELAÇÃO ENTRE Pert_r_z E Freq_inv_z")
    print("="*70)
    
    results = []
    
    for wave_name, config in WAVES_CONFIG.items():
        result = process_wave(wave_name, config)
        if result:
            results.append(result)
    
    # Resumo geral
    if results:
        print(f"\n{'='*70}")
        print("RESUMO GERAL")
        print(f"{'='*70}")
        
        summary_df = pd.DataFrame(results)
        
        print("\n[PEARSON CORRELATION]")
        print(summary_df[['wave', 'year', 'n', 'pearson_r', 'pearson_p']].to_string(index=False))
        
        print("\n[SPEARMAN CORRELATION]")
        print(summary_df[['wave', 'year', 'n', 'spearman_r', 'spearman_p']].to_string(index=False))
        
        print("\n[KENDALL CORRELATION]")
        print(summary_df[['wave', 'year', 'n', 'kendall_tau', 'kendall_p']].to_string(index=False))
        
        # Salvar resultados
        output_file = BASE_DIR / 'correlacao_pert_r_freq_inv.csv'
        summary_df.to_csv(output_file, index=False, sep=';')
        print(f"\n✓ Resultados salvos em: {output_file}")
        
        # Média das correlações (ponderada por n)
        print(f"\n[MÉDIA PONDERADA DAS CORRELAÇÕES]")
        total_n = summary_df['n'].sum()
        weighted_pearson = (summary_df['pearson_r'] * summary_df['n']).sum() / total_n
        weighted_spearman = (summary_df['spearman_r'] * summary_df['n']).sum() / total_n
        weighted_kendall = (summary_df['kendall_tau'] * summary_df['n']).sum() / total_n
        
        print(f"  Pearson (r): {weighted_pearson:.4f}")
        print(f"  Spearman (ρ): {weighted_spearman:.4f}")
        print(f"  Kendall (τ): {weighted_kendall:.4f}")


if __name__ == "__main__":
    main()







