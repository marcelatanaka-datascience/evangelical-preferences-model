"""
Script para gerar resumo comparativo de todas as medidas de correlação e confiabilidade.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path('.')

# Carrega todos os resultados
print("="*80)
print("RESUMO COMPARATIVO: MEDIDAS DE CORRELAÇÃO E CONFIABILIDADE")
print("="*80)

cronbach = pd.read_csv('cronbach_alpha_indice_religiosidade.csv')
ordinal = pd.read_csv('ordinal_alpha_indice_religiosidade.csv')
corr_pearson = pd.read_csv('correlacao_pearson_dimensoes.csv', index_col=0)
corr_spearman = pd.read_csv('correlacao_spearman_dimensoes.csv', index_col=0)
corr_kendall = pd.read_csv('correlacao_kendall_dimensoes.csv', index_col=0)
corr_polychoric = pd.read_csv('correlacao_polychoric_dimensoes.csv', index_col=0)

dimensoes = ['saliencia', 'religiosidade_individual', 'frequencia', 'organizacao']
dimensoes_labels = {
    'saliencia': 'Saliência Religiosa',
    'religiosidade_individual': 'Religiosidade Individual',
    'frequencia': 'Frequência a Serviços',
    'organizacao': 'Pertenimento a Organização'
}

# 1. Comparação de Alfa de Cronbach vs Alfa Ordinal
print("\n1. COMPARAÇÃO: ALFA DE CRONBACH vs ALFA ORDINAL")
print("-"*80)
comparison = pd.merge(
    cronbach[['wave', 'cronbach_alpha', 'interpretacao']], 
    ordinal[['wave', 'ordinal_alpha', 'interpretacao']], 
    on='wave', 
    how='outer',
    suffixes=('_cronbach', '_ordinal')
)
comparison.columns = ['Onda', 'Cronbach α', 'Interp. Cronbach', 'Ordinal α', 'Interp. Ordinal']
print(comparison.to_string(index=False))

# 2. Matriz comparativa de correlações
print("\n\n2. COMPARAÇÃO DE CORRELAÇÕES ENTRE DIMENSÕES")
print("-"*80)

pairs = [
    ('saliencia', 'frequencia', 'Saliência × Frequência'),
    ('saliencia', 'religiosidade_individual', 'Saliência × Religiosidade Individual'),
    ('saliencia', 'organizacao', 'Saliência × Pertenimento'),
    ('frequencia', 'religiosidade_individual', 'Frequência × Religiosidade Individual'),
    ('frequencia', 'organizacao', 'Frequência × Pertenimento'),
    ('religiosidade_individual', 'organizacao', 'Religiosidade × Pertenimento')
]

print(f"\n{'Par de Dimensões':<45} {'Pearson':<10} {'Spearman':<10} {'Kendall':<10} {'Polychoric':<12}")
print("-"*80)

for dim1, dim2, label in pairs:
    r_pearson = corr_pearson.loc[dim1, dim2]
    r_spearman = corr_spearman.loc[dim1, dim2]
    r_kendall = corr_kendall.loc[dim1, dim2]
    r_polychoric = corr_polychoric.loc[dim1, dim2]
    
    print(f"{label:<45} {r_pearson:>9.4f}  {r_spearman:>9.4f}  {r_kendall:>9.4f}  {r_polychoric:>11.4f}")

# 3. Interpretação
print("\n\n3. INTERPRETAÇÃO DOS RESULTADOS")
print("-"*80)
print("""
CORRELAÇÕES:
- Pearson: Assumem dados contínuos e distribuição normal (menos apropriado)
- Spearman: Baseada em ranks, apropriada para ordinais
- Kendall Tau: Também baseada em ranks, mais robusta
- Polychoric: Assume variáveis latentes contínuas subjacentes (MAIS APROPRIADO)

CONFIABILIDADE:
- Alfa de Cronbach: Assume dados contínuos (menos apropriado para ordinais)
- Alfa Ordinal: Usa correlações policóricas (MAIS APROPRIADO para ordinais)

CONCLUSÕES:
1. As correlações policóricas são geralmente mais negativas que Spearman/Kendall
2. O Alfa Ordinal ainda é baixo (0.04-0.48), confirmando baixa consistência interna
3. As dimensões são relativamente independentes, mesmo com métodos apropriados
4. Isso sugere que religiosidade é um construto multidimensional no Brasil
""")

# Salva resumo
summary_data = []
for dim1, dim2, label in pairs:
    summary_data.append({
        'par_dimensoes': label,
        'pearson': corr_pearson.loc[dim1, dim2],
        'spearman': corr_spearman.loc[dim1, dim2],
        'kendall': corr_kendall.loc[dim1, dim2],
        'polychoric': corr_polychoric.loc[dim1, dim2]
    })

summary_df = pd.DataFrame(summary_data)
summary_file = BASE_DIR / 'resumo_comparativo_correlacoes.csv'
summary_df.to_csv(summary_file, index=False, encoding='utf-8')
print(f"\nResumo comparativo salvo em: {summary_file}")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA")
print("="*80)

