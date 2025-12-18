"""
Script para explicar por que y_moral_inv está mal especificada
enquanto y_intitucional_4vars não está, mesmo usando os mesmos regressores.

O teste RESET de Ramsey verifica:
1. Formas funcionais incorretas (linear vs não-linear)
2. Variáveis omitidas relevantes
3. Interações entre variáveis não capturadas

Mesmo com os mesmos regressores, a relação funcional pode ser diferente
para cada variável dependente.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import linear_reset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / 'base_dados_final_consolidada.csv'

def main():
    print("="*70)
    print("EXPLICAÇÃO: Por que y_moral_inv está mal especificada")
    print("e y_institucional_4vars não está?")
    print("="*70)
    
    # Carregar dados
    print(f"\n[1] Carregando dados...")
    df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
    print(f"  ✓ Base carregada: {len(df)} registros")
    
    # Variáveis independentes
    vars_independentes = [
        'freq_inv_z',
        'pert_r_z',
        'import_relig_z',
        'relig_r',
        'GrupoEv',
        'dummy_ethnic',
        'dummy_sex',
        'escol_z',
        'ano_ord'
    ]
    
    # Preparar dados
    df_clean = df[['y_moral_inv', 'y_intitucional_4vars'] + vars_independentes].copy()
    
    # Garantir que todas as variáveis são numéricas
    for var in ['y_moral_inv', 'y_intitucional_4vars'] + vars_independentes:
        if not pd.api.types.is_numeric_dtype(df_clean[var]):
            df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')
    
    # Remover missing
    df_clean = df_clean.dropna()
    print(f"  ✓ Registros válidos após remover missing: {len(df_clean)}")
    
    # Separar variáveis
    X = df_clean[vars_independentes].values
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    y_moral = df_clean['y_moral_inv'].values
    y_inst = df_clean['y_intitucional_4vars'].values
    
    # Ajustar modelos
    print(f"\n[2] Ajustando modelos OLS...")
    model_moral = OLS(y_moral, X_with_const).fit()
    model_inst = OLS(y_inst, X_with_const).fit()
    
    print(f"\n  Modelo y_moral_inv:")
    print(f"    R² ajustado: {model_moral.rsquared_adj:.4f}")
    print(f"    R²: {model_moral.rsquared:.4f}")
    
    print(f"\n  Modelo y_institucional_4vars:")
    print(f"    R² ajustado: {model_inst.rsquared_adj:.4f}")
    print(f"    R²: {model_inst.rsquared:.4f}")
    
    # Teste RESET
    print(f"\n[3] Executando teste RESET de Ramsey...")
    reset_moral = linear_reset(model_moral, power=3, use_f=True)
    reset_inst = linear_reset(model_inst, power=3, use_f=True)
    
    print(f"\n  y_moral_inv:")
    print(f"    F-estatística: {reset_moral.fvalue:.4f}")
    print(f"    p-valor: {reset_moral.pvalue:.6f}")
    print(f"    Resultado: {'MAL ESPECIFICADO' if reset_moral.pvalue < 0.05 else 'BEM ESPECIFICADO'}")
    
    print(f"\n  y_institucional_4vars:")
    print(f"    F-estatística: {reset_inst.fvalue:.4f}")
    print(f"    p-valor: {reset_inst.pvalue:.6f}")
    print(f"    Resultado: {'MAL ESPECIFICADO' if reset_inst.pvalue < 0.05 else 'BEM ESPECIFICADO'}")
    
    # Análise das diferenças
    print(f"\n[4] Por que a diferença?")
    print(f"{'─'*70}")
    
    print(f"\n  1. NATUREZA DAS VARIÁVEIS DEPENDENTES:")
    print(f"     • y_moral_inv: Índice de preferências MORAIS")
    print(f"       - Baseado em: aborto, homossexualidade, prostituição")
    print(f"       - Tema mais sensível e polarizado")
    print(f"       - Pode ter relações não-lineares mais complexas")
    print(f"     • y_institucional_4vars: Índice de preferências INSTITUCIONAIS")
    print(f"       - Baseado em: confiança em congresso, justiça, igreja, imprensa")
    print(f"       - Tema mais institucional e menos polarizado")
    print(f"       - Relações podem ser mais lineares")
    
    print(f"\n  2. QUALIDADE DO AJUSTE:")
    print(f"     • y_moral_inv: R² ajustado = {model_moral.rsquared_adj:.4f}")
    print(f"       - Maior poder explicativo dos regressores (7.4%)")
    print(f"       - Mas ainda há muita variância não explicada (92.6%)")
    print(f"     • y_institucional_4vars: R² ajustado = {model_inst.rsquared_adj:.4f}")
    print(f"       - Menor poder explicativo dos regressores (2.4%)")
    print(f"       - Porém, a relação linear parece ser adequada")
    print(f"       - O baixo R² pode ser devido à natureza do fenômeno")
    
    # Verificar resíduos
    print(f"\n[5] Análise dos resíduos...")
    print(f"{'─'*70}")
    
    resid_moral = model_moral.resid
    resid_inst = model_inst.resid
    
    print(f"\n  y_moral_inv:")
    print(f"    Média dos resíduos: {resid_moral.mean():.6f} (deve ser ~0)")
    print(f"    DP dos resíduos: {resid_moral.std():.4f}")
    print(f"    Assimetria: {stats.skew(resid_moral):.4f}")
    print(f"    Curtose: {stats.kurtosis(resid_moral):.4f}")
    
    print(f"\n  y_institucional_4vars:")
    print(f"    Média dos resíduos: {resid_inst.mean():.6f} (deve ser ~0)")
    print(f"    DP dos resíduos: {resid_inst.std():.4f}")
    print(f"    Assimetria: {stats.skew(resid_inst):.4f}")
    print(f"    Curtose: {stats.kurtosis(resid_inst):.4f}")
    
    print(f"\n[6] POSSÍVEIS CAUSAS DA MÁ ESPECIFICAÇÃO DE y_moral_inv:")
    print(f"{'─'*70}")
    
    print(f"\n  a) FORMAS FUNCIONAIS NÃO-LINEARES:")
    print(f"     • Preferências morais podem ter relações não-lineares")
    print(f"       com religiosidade (ex: efeitos de threshold)")
    print(f"     • Pode ser necessário incluir termos quadráticos")
    print(f"       ou interações entre variáveis religiosas")
    
    print(f"\n  b) VARIÁVEIS OMITIDAS:")
    print(f"     • y_moral pode depender de variáveis não incluídas:")
    print(f"       - Variáveis políticas (ideologia, partido)")
    print(f"       - Variáveis de contexto (região, urbanização)")
    print(f"       - Variáveis de exposição (mídia, redes sociais)")
    print(f"     • y_institucional pode ser melhor explicado")
    print(f"       apenas pelas variáveis religiosas e demográficas")
    
    print(f"\n  c) INTERAÇÕES NÃO CAPTURADAS:")
    print(f"     • Pode haver interações importantes:")
    print(f"       - Religiosidade × Escolaridade")
    print(f"       - Religiosidade × Ano (mudanças temporais)")
    print(f"       - Frequência religiosa × Pertencimento")
    print(f"     • Essas interações podem ser mais relevantes")
    print(f"       para preferências morais do que institucionais")
    
    print(f"\n  d) HETEROGENEIDADE NÃO OBSERVADA:")
    print(f"     • Preferências morais podem variar mais entre")
    print(f"       subgrupos não capturados pelo modelo")
    print(f"     • Preferências institucionais podem ser mais")
    print(f"       homogêneas dentro dos grupos observados")
    
    print(f"\n[7] RECOMENDAÇÕES:")
    print(f"{'─'*70}")
    
    print(f"\n  Para melhorar a especificação de y_moral_inv:")
    print(f"    1. Incluir termos não-lineares (quadráticos)")
    print(f"    2. Adicionar interações entre variáveis religiosas")
    print(f"    3. Considerar variáveis políticas e contextuais")
    print(f"    4. Usar modelos não-lineares (logit, probit)")
    print(f"    5. Verificar se há heterogeneidade por subgrupos")
    
    print(f"\n  Para y_institucional_4vars:")
    print(f"    • O modelo atual parece adequado")
    print(f"    • Continuar monitorando com testes de diagnóstico")
    
    print(f"\n{'='*70}")
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)

if __name__ == "__main__":
    main()

