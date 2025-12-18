"""
Script para criar o índice Y_global que combina Y_moral e Y_institucional.

Y_global = combinação de:
- Y_moral_inv (valores morais invertidos)
- Y_institucional_4var (confiança institucional - 4 variáveis comuns)

Estratégia:
1. Carregar ambos os datasets
2. Fazer merge por wave e year
3. Criar Y_global como média dos dois índices
4. Testar consistência interna
5. Salvar dataset completo
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent

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
    
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

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
    print("="*70)
    print("CRIAÇÃO DO ÍNDICE Y_global")
    print("Y_global = combinação de Y_moral_inv + Y_institucional_4var")
    print("="*70)
    
    # Passo 1: Carregar Y_moral
    print(f"\n[PASSO 1] Carregando Y_moral_inv...")
    try:
        df_moral = pd.read_csv(BASE_DIR / 'y_moral_inv_empilhado_completo.csv', 
                               sep=';', encoding='utf-8', low_memory=False)
        print(f"  ✓ Y_moral carregado: {len(df_moral)} casos")
        print(f"  ✓ Colunas: {', '.join(df_moral.columns)}")
    except Exception as e:
        print(f"  ❌ Erro ao carregar Y_moral: {e}")
        return None
    
    # Passo 2: Carregar Y_institucional
    print(f"\n[PASSO 2] Carregando Y_institucional_4var...")
    try:
        df_inst = pd.read_csv(BASE_DIR / 'y_institucional_4var_empilhado_completo.csv', 
                             sep=';', encoding='utf-8', low_memory=False)
        print(f"  ✓ Y_institucional carregado: {len(df_inst)} casos")
        print(f"  ✓ Colunas: {', '.join(df_inst.columns)}")
    except Exception as e:
        print(f"  ❌ Erro ao carregar Y_institucional: {e}")
        return None
    
    # Passo 3: Preparar dados para merge
    print(f"\n[PASSO 3] Preparando dados para merge...")
    
    # Seleciona colunas relevantes de cada dataset
    df_moral_clean = df_moral[['Y_moral_inv', 'wave', 'year', 'year_dummy']].copy()
    df_inst_clean = df_inst[['Y_institucional_4var', 'wave', 'year', 'year_dummy']].copy()
    
    # Adiciona índice para garantir correspondência
    df_moral_clean['index'] = df_moral_clean.index
    df_inst_clean['index'] = df_inst_clean.index
    
    print(f"  ✓ Y_moral: {len(df_moral_clean)} casos")
    print(f"  ✓ Y_institucional: {len(df_inst_clean)} casos")
    
    # Passo 4: Fazer merge
    print(f"\n[PASSO 4] Fazendo merge dos datasets...")
    
    # Merge por wave, year e índice
    df_combined = pd.merge(df_moral_clean, df_inst_clean, 
                          on=['wave', 'year', 'year_dummy', 'index'],
                          how='inner', suffixes=('', '_inst'))
    
    print(f"  ✓ Merge concluído: {len(df_combined)} casos combinados")
    
    # Verifica se há casos perdidos
    if len(df_combined) < len(df_moral_clean):
        print(f"  ⚠️  {len(df_moral_clean) - len(df_combined)} casos de Y_moral não foram combinados")
    if len(df_combined) < len(df_inst_clean):
        print(f"  ⚠️  {len(df_inst_clean) - len(df_combined)} casos de Y_institucional não foram combinados")
    
    # Passo 5: Criar Y_global
    print(f"\n[PASSO 5] Criando índice Y_global...")
    
    # Y_global = média de Y_moral_inv e Y_institucional_4var
    df_combined['Y_global'] = df_combined[['Y_moral_inv', 'Y_institucional_4var']].mean(axis=1, skipna=False)
    
    valid_cases = df_combined['Y_global'].notna().sum()
    print(f"  ✓ Y_global criado: média={df_combined['Y_global'].mean():.6f}, DP={df_combined['Y_global'].std():.6f}")
    print(f"  ✓ Valores válidos: {valid_cases} de {len(df_combined)}")
    
    # Estatísticas descritivas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS DOS ÍNDICES:")
    print(f"{'─'*70}\n")
    
    for var in ['Y_moral_inv', 'Y_institucional_4var', 'Y_global']:
        valid = df_combined[var].dropna()
        if len(valid) > 0:
            print(f"{var}:")
            print(f"  N: {len(valid)}, Média: {valid.mean():.6f}, DP: {valid.std():.6f}")
            print(f"  Mín: {valid.min():.6f}, Máx: {valid.max():.6f}")
    
    # Passo 6: Testar consistência interna
    print(f"\n[PASSO 6] Testando consistência interna do Y_global...")
    print(f"{'─'*70}\n")
    
    # Alfa de Cronbach usando os dois índices como itens
    indices_vars = ['Y_moral_inv', 'Y_institucional_4var']
    alpha_global = cronbach_alpha(df_combined, indices_vars)
    
    if pd.isna(alpha_global):
        print("  ❌ Não foi possível calcular o Alfa de Cronbach.")
    else:
        interp = interpret_alpha(alpha_global)
        n_valid = df_combined[indices_vars].dropna().shape[0]
        print(f"  ✓ Alfa de Cronbach: {alpha_global:.4f} ({interp})")
        print(f"  ✓ Casos válidos: {n_valid}")
        print(f"  ✓ Índices combinados: Y_moral_inv + Y_institucional_4var")
    
    # Correlação entre os índices
    print(f"\n{'─'*70}")
    print("CORRELAÇÃO ENTRE OS ÍNDICES:")
    print(f"{'─'*70}\n")
    
    corr = df_combined['Y_moral_inv'].corr(df_combined['Y_institucional_4var'])
    print(f"Correlação Y_moral_inv vs Y_institucional_4var: r = {corr:.4f}")
    
    # Alfa por wave
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH POR ONDA:")
    print(f"{'─'*70}\n")
    
    results_by_wave = []
    
    for wave_name in sorted(df_combined['wave'].unique()):
        wave_df = df_combined[df_combined['wave'] == wave_name]
        alpha_wave = cronbach_alpha(wave_df, indices_vars)
        n_wave = wave_df[indices_vars].dropna().shape[0]
        year = wave_df['year'].iloc[0] if len(wave_df) > 0 else None
        
        interp = interpret_alpha(alpha_wave)
        print(f"{wave_name:20s} ({year}): α = {alpha_wave:.4f} ({interp}), N = {n_wave}")
        
        results_by_wave.append({
            'wave': wave_name,
            'year': year,
            'cronbach_alpha': alpha_wave,
            'interpretacao': interp,
            'n_casos_validos': n_wave
        })
    
    # Estatísticas por ano
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DO Y_global POR ANO:")
    print(f"{'─'*70}\n")
    
    stats_by_year = df_combined.groupby('year')['Y_global'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(4)
    stats_by_year.columns = ['N', 'Média', 'DP', 'Mín', 'Máx']
    print(stats_by_year)
    
    # Passo 7: Preparar dataset final
    print(f"\n[PASSO 7] Preparando dataset final...")
    
    # Seleciona colunas finais
    df_final = df_combined[[
        'Y_moral_inv',
        'Y_institucional_4var', 
        'Y_global',
        'wave',
        'year',
        'year_dummy'
    ]].copy()
    
    # Adiciona também as variáveis individuais se necessário
    # (opcional - pode adicionar depois se quiser)
    
    # Passo 8: Salvar
    print(f"\n[PASSO 8] Salvando resultados...")
    
    # Dataset completo
    output_file = BASE_DIR / 'y_global_completo.csv'
    df_final.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"  ✓ Dataset completo salvo: {output_file}")
    print(f"  ✓ Total de casos: {len(df_final)}")
    print(f"  ✓ Colunas: {', '.join(df_final.columns)}")
    
    # Resumo
    summary_data = results_by_wave.copy()
    summary_data.insert(0, {
        'wave': 'Todas as ondas',
        'year': '1991-2018',
        'cronbach_alpha': alpha_global,
        'interpretacao': interpret_alpha(alpha_global),
        'n_casos_validos': n_valid if not pd.isna(alpha_global) else 0
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = BASE_DIR / 'y_global_resumo.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ Resumo salvo: {summary_file}")
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*70)
    print(f"\nY_global criado com sucesso!")
    print(f"  • Combina Y_moral_inv e Y_institucional_4var")
    print(f"  • Alfa de Cronbach: {alpha_global:.4f} ({interpret_alpha(alpha_global)})")
    print(f"  • Correlação entre índices: r = {corr:.4f}")
    
    return df_final

if __name__ == "__main__":
    main()








