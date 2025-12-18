"""
Script para testar multicolinearidade entre variáveis independentes
usando VIF (Variance Inflation Factor).

Variáveis testadas:
- freq_inv_z (freq_in_z mencionada pelo usuário)
- pert_r_z
- import_relig_z
- relig_r
- GrupoEv
- dummy_ethnic
- dummy_sex
- escol_z
- ano_ord (anor_ord mencionada pelo usuário)
- ameaca_tipo_inst (ameaca_tipo mencionada pelo usuário)
- ameaca_tipo_moral

Interpretação do VIF:
- VIF < 5: Multicolinearidade baixa (aceitável)
- 5 <= VIF < 10: Multicolinearidade moderada (atenção)
- VIF >= 10: Multicolinearidade alta (problema sério)
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / 'base_dados_final_consolidada.csv'

def calculate_vif(data, variables):
    """
    Calcula o VIF para cada variável.
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame com os dados
    variables : list
        Lista de variáveis para calcular VIF
    
    Returns:
    --------
    DataFrame com VIF para cada variável
    """
    # Preparar dados: remover missing e garantir que são numéricos
    data_clean = data[variables].copy()
    data_clean = data_clean.dropna()
    
    # Verificar se há dados suficientes
    if len(data_clean) < len(variables) + 1:
        print(f"⚠️  Atenção: Dados insuficientes para calcular VIF")
        print(f"   Registros válidos: {len(data_clean)}, Variáveis: {len(variables)}")
        return None
    
    # Garantir que todas as variáveis são numéricas
    for var in variables:
        if not pd.api.types.is_numeric_dtype(data_clean[var]):
            print(f"⚠️  Convertendo {var} para numérico...")
            data_clean[var] = pd.to_numeric(data_clean[var], errors='coerce')
    
    # Remover missing após conversão
    data_clean = data_clean.dropna()
    
    # Adicionar constante para regressão
    X = data_clean[variables].values
    
    # Calcular VIF para cada variável
    vif_data = []
    for i, var in enumerate(variables):
        try:
            vif = variance_inflation_factor(X, i)
            vif_data.append({
                'Variável': var,
                'VIF': vif,
                'Interpretação': interpret_vif(vif)
            })
        except Exception as e:
            print(f"⚠️  Erro ao calcular VIF para {var}: {e}")
            vif_data.append({
                'Variável': var,
                'VIF': np.nan,
                'Interpretação': 'Erro no cálculo'
            })
    
    vif_df = pd.DataFrame(vif_data)
    return vif_df, len(data_clean)

def interpret_vif(vif):
    """Interpreta o valor do VIF."""
    if pd.isna(vif):
        return "N/A"
    elif vif < 5:
        return "Baixa (aceitável)"
    elif vif < 10:
        return "Moderada (atenção)"
    else:
        return "Alta (problema sério)"

def calculate_correlation_matrix(data, variables):
    """Calcula matriz de correlação entre as variáveis."""
    data_clean = data[variables].copy()
    
    # Garantir que todas as variáveis são numéricas
    for var in variables:
        if not pd.api.types.is_numeric_dtype(data_clean[var]):
            data_clean[var] = pd.to_numeric(data_clean[var], errors='coerce')
    
    # Remover missing
    data_clean = data_clean.dropna()
    
    # Calcular correlação
    corr_matrix = data_clean[variables].corr()
    return corr_matrix, len(data_clean)

def main():
    """Função principal."""
    print("="*70)
    print("TESTE DE MULTICOLINEARIDADE - VIF (Variance Inflation Factor)")
    print("="*70)
    
    # Carregar base consolidada
    print(f"\n[1] Carregando base consolidada...")
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
        print(f"  ✓ Base carregada: {len(df)} registros, {len(df.columns)} colunas")
    except Exception as e:
        print(f"  ❌ Erro ao carregar base: {e}")
        return
    
    # Definir variáveis independentes
    # Ajustando nomes conforme mencionado pelo usuário
    # Removido: 'ameaca_tipo_inst' (multicolinearidade alta)
    variables = [
        'freq_inv_z',      # freq_in_z mencionada
        'pert_r_z',
        'import_relig_z',
        'relig_r',
        'GrupoEv',
        'dummy_ethnic',
        'dummy_sex',
        'escol_z',
        'ano_ord',         # anor_ord mencionada
        'ameaca_tipo_moral',
        'ameaca_tipo_inst'
    ]
    
    # Verificar quais variáveis existem
    print(f"\n[2] Verificando variáveis...")
    missing_vars = []
    available_vars = []
    
    for var in variables:
        if var in df.columns:
            available_vars.append(var)
            missing_count = df[var].isna().sum()
            total_count = len(df)
            pct_missing = (missing_count / total_count) * 100
            print(f"  ✓ {var:20s}: {total_count - missing_count:5d} válidos ({100-pct_missing:5.2f}%)")
        else:
            missing_vars.append(var)
            print(f"  ❌ {var:20s}: NÃO ENCONTRADA")
    
    if missing_vars:
        print(f"\n⚠️  Variáveis não encontradas: {', '.join(missing_vars)}")
        print(f"   Continuando com as variáveis disponíveis...")
    
    if not available_vars:
        print(f"\n❌ Nenhuma variável disponível para análise!")
        return
    
    # Estatísticas descritivas
    print(f"\n[3] Estatísticas descritivas das variáveis...")
    print(f"{'─'*70}")
    stats = df[available_vars].describe().T
    stats['missing'] = df[available_vars].isna().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    print(stats.round(4))
    
    # Calcular VIF
    print(f"\n[4] Calculando VIF (Variance Inflation Factor)...")
    print(f"{'─'*70}")
    
    vif_df, n_valid = calculate_vif(df, available_vars)
    
    if vif_df is not None:
        print(f"\n  Registros utilizados (sem missing): {n_valid}")
        print(f"\n  Resultados do VIF:")
        print(f"{'─'*70}")
        print(vif_df.to_string(index=False))
        
        # Resumo
        print(f"\n[5] Resumo dos resultados...")
        print(f"{'─'*70}")
        
        vif_values = vif_df['VIF'].dropna()
        if len(vif_values) > 0:
            print(f"  VIF mínimo: {vif_values.min():.4f}")
            print(f"  VIF máximo: {vif_values.max():.4f}")
            print(f"  VIF médio:  {vif_values.mean():.4f}")
            print(f"  VIF mediano: {vif_values.median():.4f}")
            
            # Contar por categoria
            vif_baixa = (vif_values < 5).sum()
            vif_moderada = ((vif_values >= 5) & (vif_values < 10)).sum()
            vif_alta = (vif_values >= 10).sum()
            
            print(f"\n  Distribuição:")
            print(f"    Baixa (VIF < 5):     {vif_baixa:2d} variáveis")
            print(f"    Moderada (5-10):     {vif_moderada:2d} variáveis")
            print(f"    Alta (VIF >= 10):    {vif_alta:2d} variáveis")
            
            # Alertas
            print(f"\n[6] Alertas de multicolinearidade...")
            print(f"{'─'*70}")
            
            problemas = vif_df[vif_df['VIF'] >= 10]
            if len(problemas) > 0:
                print(f"  ⚠️  VARIÁVEIS COM MULTICOLINEARIDADE ALTA (VIF >= 10):")
                for _, row in problemas.iterrows():
                    print(f"     - {row['Variável']:20s}: VIF = {row['VIF']:.4f}")
            else:
                print(f"  ✓ Nenhuma variável com multicolinearidade alta")
            
            atencao = vif_df[(vif_df['VIF'] >= 5) & (vif_df['VIF'] < 10)]
            if len(atencao) > 0:
                print(f"\n  ⚠️  VARIÁVEIS COM MULTICOLINEARIDADE MODERADA (5 <= VIF < 10):")
                for _, row in atencao.iterrows():
                    print(f"     - {row['Variável']:20s}: VIF = {row['VIF']:.4f}")
            else:
                print(f"  ✓ Nenhuma variável com multicolinearidade moderada")
        else:
            print(f"  ❌ Não foi possível calcular VIF para nenhuma variável")
    else:
        print(f"  ❌ Erro ao calcular VIF")
    
    # Matriz de correlação
    print(f"\n[7] Matriz de correlação entre variáveis...")
    print(f"{'─'*70}")
    
    corr_matrix, n_valid_corr = calculate_correlation_matrix(df, available_vars)
    
    if corr_matrix is not None:
        print(f"\n  Registros utilizados (sem missing): {n_valid_corr}")
        print(f"\n  Matriz de correlação (valores absolutos >= 0.7 destacados):")
        print(f"{'─'*70}")
        
        # Formatar matriz para exibição
        corr_display = corr_matrix.round(3)
        
        # Destacar correlações altas
        print("\n  Correlações altas (|r| >= 0.7):")
        high_corr = []
        for i in range(len(corr_display.columns)):
            for j in range(i+1, len(corr_display.columns)):
                corr_val = corr_display.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    high_corr.append((
                        corr_display.columns[i],
                        corr_display.columns[j],
                        corr_val
                    ))
        
        if high_corr:
            for var1, var2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                print(f"    {var1:20s} <-> {var2:20s}: r = {corr:6.3f}")
        else:
            print(f"    Nenhuma correlação alta encontrada")
        
        # Salvar matriz completa
        output_corr = BASE_DIR / 'matriz_correlacao_multicolinearidade.csv'
        corr_matrix.to_csv(output_corr, sep=';', encoding='utf-8-sig')
        print(f"\n  ✓ Matriz de correlação completa salva em: {output_corr}")
    else:
        print(f"  ❌ Erro ao calcular matriz de correlação")
    
    # Salvar resultados do VIF
    if vif_df is not None:
        output_vif = BASE_DIR / 'resultados_vif_multicolinearidade.csv'
        vif_df.to_csv(output_vif, sep=';', index=False, encoding='utf-8-sig')
        print(f"\n  ✓ Resultados do VIF salvos em: {output_vif}")
    
    print(f"\n{'='*70}")
    print("ANÁLISE DE MULTICOLINEARIDADE CONCLUÍDA!")
    print("="*70)
    
    # Recomendações
    print(f"\n[8] Recomendações...")
    print(f"{'─'*70}")
    
    if vif_df is not None:
        vif_values = vif_df['VIF'].dropna()
        if len(vif_values) > 0:
            if (vif_values >= 10).any():
                print(f"  ⚠️  Existem variáveis com multicolinearidade alta (VIF >= 10).")
                print(f"     Considere remover ou combinar essas variáveis.")
            elif (vif_values >= 5).any():
                print(f"  ⚠️  Existem variáveis com multicolinearidade moderada (5 <= VIF < 10).")
                print(f"     Monitore essas variáveis e considere revisar o modelo.")
            else:
                print(f"  ✓ Todas as variáveis apresentam multicolinearidade baixa (VIF < 5).")
                print(f"     O modelo está adequado em termos de multicolinearidade.")

if __name__ == "__main__":
    main()

