"""
Script para criar o índice Y_global diretamente a partir de todas as variáveis
padronizadas, sem usar os índices intermediários Y_moral e Y_institucional.

Y_global = média de todas as variáveis padronizadas:
- Valores morais: aborto_z, homossex_z, prost_z
- Confiança institucional: congresso_z, justica_z, igreja_z, imprensa_z
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
    print("CRIAÇÃO DO ÍNDICE Y_global (MÉTODO DIRETO)")
    print("Y_global = média de todas as variáveis padronizadas")
    print("="*70)
    
    # Passo 1: Carregar Y_moral (para pegar variáveis morais)
    print(f"\n[PASSO 1] Carregando variáveis de Y_moral...")
    try:
        df_moral = pd.read_csv(BASE_DIR / 'y_moral_inv_empilhado_completo.csv', 
                               sep=';', encoding='utf-8', low_memory=False)
        print(f"  ✓ Y_moral carregado: {len(df_moral)} casos")
    except Exception as e:
        print(f"  ❌ Erro ao carregar Y_moral: {e}")
        return None
    
    # Passo 2: Carregar Y_institucional (para pegar variáveis institucionais)
    print(f"\n[PASSO 2] Carregando variáveis de Y_institucional...")
    try:
        df_inst = pd.read_csv(BASE_DIR / 'y_institucional_4var_empilhado_completo.csv', 
                             sep=';', encoding='utf-8', low_memory=False)
        print(f"  ✓ Y_institucional carregado: {len(df_inst)} casos")
    except Exception as e:
        print(f"  ❌ Erro ao carregar Y_institucional: {e}")
        return None
    
    # Passo 3: Combinar todas as variáveis
    print(f"\n[PASSO 3] Combinando todas as variáveis...")
    
    # Variáveis morais
    vars_morais = ['aborto_z', 'homossex_z', 'prost_z']
    
    # Variáveis institucionais
    vars_institucionais = ['congresso_z', 'justica_z', 'igreja_z', 'imprensa_z']
    
    # Todas as variáveis
    todas_vars = vars_morais + vars_institucionais
    
    print(f"  ✓ Variáveis morais: {vars_morais}")
    print(f"  ✓ Variáveis institucionais: {vars_institucionais}")
    print(f"  ✓ Total: {len(todas_vars)} variáveis")
    
    # Seleciona colunas relevantes
    df_moral_vars = df_moral[['wave', 'year', 'year_dummy'] + vars_morais].copy()
    df_inst_vars = df_inst[['wave', 'year', 'year_dummy'] + vars_institucionais].copy()
    
    # Adiciona índice para merge
    df_moral_vars['index'] = df_moral_vars.index
    df_inst_vars['index'] = df_inst_vars.index
    
    # Merge
    df_combined = pd.merge(df_moral_vars, df_inst_vars,
                          on=['wave', 'year', 'year_dummy', 'index'],
                          how='inner', suffixes=('', '_inst'))
    
    print(f"  ✓ Merge concluído: {len(df_combined)} casos")
    
    # Passo 4: Criar Y_global diretamente
    print(f"\n[PASSO 4] Criando Y_global diretamente a partir de todas as variáveis...")
    
    # Verifica quais variáveis estão disponíveis
    vars_disponiveis = [v for v in todas_vars if v in df_combined.columns]
    
    if len(vars_disponiveis) < len(todas_vars):
        print(f"  ⚠️  Apenas {len(vars_disponiveis)} de {len(todas_vars)} variáveis disponíveis")
        print(f"  Variáveis disponíveis: {vars_disponiveis}")
    
    # Calcula Y_global como média de todas as variáveis
    df_combined['Y_global'] = df_combined[vars_disponiveis].mean(axis=1, skipna=False)
    
    valid_cases = df_combined['Y_global'].notna().sum()
    print(f"  ✓ Y_global criado: média={df_combined['Y_global'].mean():.6f}, DP={df_combined['Y_global'].std():.6f}")
    print(f"  ✓ Valores válidos: {valid_cases} de {len(df_combined)}")
    print(f"  ✓ Variáveis utilizadas: {len(vars_disponiveis)}")
    
    # Passo 5: Testar consistência interna
    print(f"\n[PASSO 5] Testando consistência interna do Y_global...")
    print(f"{'─'*70}\n")
    
    # Alfa de Cronbach usando todas as variáveis individuais
    alpha_global = cronbach_alpha(df_combined, vars_disponiveis)
    
    if pd.isna(alpha_global):
        print("  ❌ Não foi possível calcular o Alfa de Cronbach.")
    else:
        interp = interpret_alpha(alpha_global)
        n_valid = df_combined[vars_disponiveis].dropna().shape[0]
        print(f"  ✓ Alfa de Cronbach: {alpha_global:.4f} ({interp})")
        print(f"  ✓ Casos válidos: {n_valid}")
        print(f"  ✓ Variáveis: {len(vars_disponiveis)}")
    
    # Matriz de correlação
    print(f"\n{'─'*70}")
    print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
    print(f"{'─'*70}\n")
    
    corr_matrix = df_combined[vars_disponiveis].corr(method='pearson')
    print(corr_matrix.round(4))
    
    # Correlações médias
    print(f"\n{'─'*70}")
    print("CORRELAÇÕES MÉDIAS:")
    print(f"{'─'*70}\n")
    
    # Correlação média entre variáveis morais
    corr_morais = corr_matrix.loc[vars_morais, vars_morais].values
    corr_morais_mean = corr_morais[np.triu_indices(len(vars_morais), k=1)].mean()
    print(f"Correlação média entre variáveis morais: {corr_morais_mean:.4f}")
    
    # Correlação média entre variáveis institucionais
    corr_inst = corr_matrix.loc[vars_institucionais, vars_institucionais].values
    corr_inst_mean = corr_inst[np.triu_indices(len(vars_institucionais), k=1)].mean()
    print(f"Correlação média entre variáveis institucionais: {corr_inst_mean:.4f}")
    
    # Correlação média entre grupos
    corr_between = corr_matrix.loc[vars_morais, vars_institucionais].values.mean()
    print(f"Correlação média entre grupos (moral vs institucional): {corr_between:.4f}")
    
    # Alfa por wave
    print(f"\n{'─'*70}")
    print("ALFA DE CRONBACH POR ONDA:")
    print(f"{'─'*70}\n")
    
    results_by_wave = []
    
    for wave_name in sorted(df_combined['wave'].unique()):
        wave_df = df_combined[df_combined['wave'] == wave_name]
        alpha_wave = cronbach_alpha(wave_df, vars_disponiveis)
        n_wave = wave_df[vars_disponiveis].dropna().shape[0]
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
    
    # Estatísticas descritivas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(f"{'─'*70}\n")
    
    for var in vars_disponiveis + ['Y_global']:
        valid = df_combined[var].dropna()
        if len(valid) > 0:
            print(f"{var}:")
            print(f"  N: {len(valid)}, Média: {valid.mean():.6f}, DP: {valid.std():.6f}")
            print(f"  Mín: {valid.min():.6f}, Máx: {valid.max():.6f}")
    
    # Passo 6: Salvar
    print(f"\n[PASSO 6] Salvando resultados...")
    
    # Dataset final com todas as variáveis
    df_final = df_combined[['wave', 'year', 'year_dummy'] + vars_disponiveis + ['Y_global']].copy()
    
    output_file = BASE_DIR / 'y_global_direto_completo.csv'
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
    summary_file = BASE_DIR / 'y_global_direto_resumo.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ Resumo salvo: {summary_file}")
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*70)
    print(f"\nY_global criado diretamente a partir de {len(vars_disponiveis)} variáveis!")
    print(f"  • Alfa de Cronbach: {alpha_global:.4f} ({interpret_alpha(alpha_global)})")
    print(f"  • Variáveis morais: {len(vars_morais)}")
    print(f"  • Variáveis institucionais: {len(vars_institucionais)}")
    
    return df_final

if __name__ == "__main__":
    main()








