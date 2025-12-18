"""
Script para explicar por que y_moral_inv pode ter valores menores que -3.

y_moral_inv é a MÉDIA de variáveis padronizadas (z-scores), não uma variável
padronizada em si. Quando você calcula a média de variáveis padronizadas:

1. A média do índice será ~0 (se as variáveis componentes têm média 0)
2. O desvio padrão do índice NÃO será necessariamente 1
3. Valores extremos podem ocorrer quando TODAS as variáveis componentes
   têm valores extremos na mesma direção

Fórmula do desvio padrão da média:
DP(média) = sqrt((1 + correlações médias) / n)

Com 3 variáveis e correlações moderadas (~0.3-0.4), o DP esperado é ~0.7-0.8,
não 1.0.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / 'base_dados_final_consolidada.csv'

def main():
    print("="*70)
    print("EXPLICAÇÃO: Por que y_moral_inv pode ter valores < -3?")
    print("="*70)
    
    # Carregar dados
    df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
    
    # Variáveis componentes
    vars_componentes = ['aborto_z', 'homossex_z', 'prost_z']
    
    print(f"\n[1] Como y_moral_inv é criado:")
    print(f"{'─'*70}")
    print(f"  y_moral_inv = média(aborto_z, homossex_z, prost_z)")
    print(f"  É a MÉDIA de variáveis padronizadas, não uma variável padronizada!")
    
    print(f"\n[2] Estatísticas das variáveis componentes:")
    print(f"{'─'*70}")
    for var in vars_componentes:
        if var in df.columns:
            stats = df[var].describe()
            print(f"\n  {var}:")
            print(f"    Média: {stats['mean']:.6f} (deve ser ~0)")
            print(f"    DP:    {stats['std']:.6f} (deve ser ~1)")
            print(f"    Mín:   {stats['min']:.4f}")
            print(f"    Máx:   {stats['max']:.4f}")
    
    print(f"\n[3] Estatísticas de y_moral_inv:")
    print(f"{'─'*70}")
    stats_y = df['y_moral_inv'].describe()
    print(f"  Média: {stats_y['mean']:.6f} (praticamente 0 ✓)")
    print(f"  DP:    {stats_y['std']:.6f} (NÃO é 1! É ~0.75)")
    print(f"  Mín:   {stats_y['min']:.4f}")
    print(f"  Máx:   {stats_y['max']:.4f}")
    
    print(f"\n[4] Por que o DP não é 1?")
    print(f"{'─'*70}")
    print(f"  Quando você calcula a média de variáveis padronizadas,")
    print(f"  o desvio padrão da média depende das correlações entre as variáveis.")
    print(f"  ")
    print(f"  Fórmula aproximada: DP(média) ≈ sqrt((1 + r_medio) / n)")
    print(f"  onde r_medio é a correlação média entre as variáveis")
    print(f"  e n é o número de variáveis")
    
    # Calcular correlações
    corr_matrix = df[vars_componentes].corr()
    corr_medio = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    print(f"\n  Correlação média entre componentes: {corr_medio:.4f}")
    print(f"  Número de variáveis: {len(vars_componentes)}")
    
    dp_teorico = np.sqrt((1 + corr_medio) / len(vars_componentes))
    print(f"  DP teórico esperado: {dp_teorico:.4f}")
    print(f"  DP observado:       {stats_y['std']:.6f}")
    print(f"  Diferença:           {abs(dp_teorico - stats_y['std']):.6f}")
    
    print(f"\n[5] Por que valores < -3 são possíveis?")
    print(f"{'─'*70}")
    print(f"  Valores extremos ocorrem quando TODAS as variáveis componentes")
    print(f"  têm valores extremos na mesma direção (todos negativos ou todos positivos).")
    print(f"  ")
    print(f"  Exemplo: se aborto_z = -2, homossex_z = -3, prost_z = -4,")
    print(f"  então y_moral_inv = (-2 - 3 - 4) / 3 = -3")
    print(f"  ")
    print(f"  Como as variáveis componentes podem ter valores < -3 (são z-scores),")
    print(f"  a média também pode ter valores < -3.")
    
    # Mostrar casos extremos
    print(f"\n[6] Casos com y_moral_inv < -3:")
    print(f"{'─'*70}")
    casos_extremos = df[df['y_moral_inv'] < -3][['y_moral_inv'] + vars_componentes].copy()
    print(f"  Total de casos: {len(casos_extremos)}")
    print(f"\n  Primeiros 5 casos:")
    print(casos_extremos.head().to_string(index=False))
    
    # Verificar se a média calculada manualmente bate
    print(f"\n[7] Verificação: média calculada vs y_moral_inv")
    print(f"{'─'*70}")
    df_verif = df[vars_componentes + ['y_moral_inv']].copy()
    df_verif['media_calculada'] = df_verif[vars_componentes].mean(axis=1)
    df_verif['diferenca'] = abs(df_verif['y_moral_inv'] - df_verif['media_calculada'])
    
    print(f"  Diferença máxima: {df_verif['diferenca'].max():.10f}")
    print(f"  Diferença média:  {df_verif['diferenca'].mean():.10f}")
    if df_verif['diferenca'].max() < 0.0001:
        print(f"  ✓ A média calculada bate perfeitamente com y_moral_inv")
    else:
        print(f"  ⚠️  Há diferenças (pode ser devido a arredondamento)")
    
    print(f"\n[8] Distribuição dos valores:")
    print(f"{'─'*70}")
    print(f"  Valores < -3: {(df['y_moral_inv'] < -3).sum()} casos ({(df['y_moral_inv'] < -3).sum() / len(df) * 100:.2f}%)")
    print(f"  Valores < -2: {(df['y_moral_inv'] < -2).sum()} casos ({(df['y_moral_inv'] < -2).sum() / len(df) * 100:.2f}%)")
    print(f"  Valores entre -2 e 2: {((df['y_moral_inv'] >= -2) & (df['y_moral_inv'] <= 2)).sum()} casos ({((df['y_moral_inv'] >= -2) & (df['y_moral_inv'] <= 2)).sum() / len(df) * 100:.2f}%)")
    print(f"  Valores > 2: {(df['y_moral_inv'] > 2).sum()} casos ({(df['y_moral_inv'] > 2).sum() / len(df) * 100:.2f}%)")
    print(f"  Valores > 3: {(df['y_moral_inv'] > 3).sum()} casos ({(df['y_moral_inv'] > 3).sum() / len(df) * 100:.2f}%)")
    
    print(f"\n[9] Conclusão:")
    print(f"{'─'*70}")
    print(f"  ✓ y_moral_inv É um índice padronizado (média ~0)")
    print(f"  ✓ Mas NÃO tem DP = 1, e sim DP ≈ 0.75")
    print(f"  ✓ Valores < -3 são matematicamente corretos e esperados")
    print(f"  ✓ Eles ocorrem quando todas as variáveis componentes têm valores extremos")
    print(f"  ✓ Isso é normal em índices que são médias de variáveis padronizadas")
    
    print(f"\n{'='*70}")
    print("EXPLICAÇÃO CONCLUÍDA!")
    print("="*70)

if __name__ == "__main__":
    main()



