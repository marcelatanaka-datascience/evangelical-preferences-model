#!/usr/bin/env python3
"""
Análise dos respondentes da WVS Wave 7 Brazil separados por religião
Comparação das questões Q20 e Q21 entre evangélicos, católicos e outros
"""

import pandas as pd
import numpy as np
from savReaderWriter import SavReader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_wvs_data(sav_file_path):
    """Carrega dados do arquivo .sav usando savReaderWriter"""
    print(f"Carregando dados de: {sav_file_path}")
    
    with SavReader(sav_file_path, returnHeader=True, ioUtf8=True) as reader:
        header = next(reader)
        data = list(reader)
    
    df = pd.DataFrame(data, columns=header)
    print(f"Dados carregados com sucesso! Shape: {df.shape}")
    return df

def explore_variables(df):
    """Explora as variáveis para encontrar religião, Q20 e Q21"""
    print("\n" + "="*60)
    print("EXPLORANDO VARIÁVEIS")
    print("="*60)
    
    # Procurar variáveis de religião
    religiao_vars = [col for col in df.columns if 'relig' in col.lower() or 'Q289' in col.upper() or 'Q289A' in col.upper()]
    print(f"\nVariáveis de religião encontradas: {religiao_vars}")
    
    # Procurar Q20 e Q21
    q20_vars = [col for col in df.columns if 'Q20' in col.upper() and 'Q20' == col.upper()[:3]]
    q21_vars = [col for col in df.columns if 'Q21' in col.upper() and 'Q21' == col.upper()[:3]]
    
    # Também procurar variações
    if not q20_vars:
        q20_vars = [col for col in df.columns if col.upper().startswith('Q20')]
    if not q21_vars:
        q21_vars = [col for col in df.columns if col.upper().startswith('Q21')]
    
    print(f"Variáveis Q20 encontradas: {q20_vars}")
    print(f"Variáveis Q21 encontradas: {q21_vars}")
    
    # Mostrar algumas variáveis que começam com Q
    q_vars = [col for col in df.columns if col.upper().startswith('Q')]
    print(f"\nTotal de variáveis que começam com Q: {len(q_vars)}")
    print(f"Primeiras 20: {sorted(q_vars)[:20]}")
    
    return religiao_vars, q20_vars, q21_vars

def map_religion(df, religiao_var):
    """Mapeia valores de religião para categorias: Evangélicos, Católicos, Outros"""
    print(f"\n" + "="*60)
    print(f"ANÁLISE DA VARIÁVEL DE RELIGIÃO: {religiao_var}")
    print("="*60)
    
    # Ver valores únicos
    valores_unicos = df[religiao_var].value_counts().sort_index()
    print("\nValores únicos e suas frequências:")
    for valor, count in valores_unicos.items():
        print(f"  {valor}: {count}")
    
    # Mapear para categorias
    # No WVS, geralmente:
    # 1 = Católico
    # 2 = Protestante/Evangélico
    # Outros valores = Outras religiões
    
    def map_religion_value(val):
        if pd.isna(val):
            return 'Outros'
        val = float(val)
        if val == 1.0:
            return 'Católicos'
        elif val == 2.0:
            return 'Evangélicos'
        else:
            return 'Outros'
    
    df['religiao_grupo'] = df[religiao_var].apply(map_religion_value)
    
    print("\nDistribuição por grupo religioso:")
    print(df['religiao_grupo'].value_counts())
    print("\nPercentual por grupo religioso:")
    print(df['religiao_grupo'].value_counts(normalize=True) * 100)
    
    return df

def analyze_question(df, question_var, question_name):
    """Analisa uma questão específica por grupo religioso"""
    print(f"\n" + "="*60)
    print(f"ANÁLISE DA {question_name}: {question_var}")
    print("="*60)
    
    # Verificar se a variável existe
    if question_var not in df.columns:
        print(f"AVISO: Variável {question_var} não encontrada!")
        return None
    
    # Remover valores missing
    df_clean = df[[question_var, 'religiao_grupo']].dropna()
    
    # Tabela de contingência
    crosstab = pd.crosstab(df_clean['religiao_grupo'], df_clean[question_var], margins=True)
    print(f"\nTabela de contingência ({question_name}):")
    print(crosstab)
    
    # Percentuais por grupo
    crosstab_pct = pd.crosstab(df_clean['religiao_grupo'], df_clean[question_var], normalize='index') * 100
    print(f"\nPercentuais por grupo religioso ({question_name}):")
    print(crosstab_pct.round(2))
    
    # Teste qui-quadrado
    crosstab_test = pd.crosstab(df_clean['religiao_grupo'], df_clean[question_var])
    chi2, p_value, dof, expected = chi2_contingency(crosstab_test)
    print(f"\nTeste Qui-quadrado:")
    print(f"  Chi-quadrado: {chi2:.4f}")
    print(f"  p-valor: {p_value:.4f}")
    print(f"  Graus de liberdade: {dof}")
    if p_value < 0.05:
        print(f"  Resultado: Diferenças estatisticamente significativas (p < 0.05)")
    else:
        print(f"  Resultado: Diferenças não são estatisticamente significativas (p >= 0.05)")
    
    # Estatísticas descritivas por grupo
    print(f"\nEstatísticas descritivas por grupo:")
    stats = df_clean.groupby('religiao_grupo')[question_var].agg(['count', 'mean', 'std', 'min', 'max'])
    print(stats)
    
    return {
        'crosstab': crosstab,
        'crosstab_pct': crosstab_pct,
        'chi2': chi2,
        'p_value': p_value,
        'stats': stats
    }

def create_visualizations(df, q20_var, q21_var, q20_analysis, q21_analysis):
    """Cria visualizações das análises"""
    print("\n" + "="*60)
    print("CRIANDO VISUALIZAÇÕES")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico 1: Distribuição Q20 por grupo religioso (barras)
    if q20_analysis:
        ax1 = axes[0, 0]
        q20_analysis['crosstab_pct'].T.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Distribuição das Respostas Q20 por Grupo Religioso (%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Resposta Q20', fontsize=10)
        ax1.set_ylabel('Percentual (%)', fontsize=10)
        ax1.legend(title='Grupo Religioso')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Distribuição Q21 por grupo religioso (barras)
    if q21_analysis:
        ax2 = axes[0, 1]
        q21_analysis['crosstab_pct'].T.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Distribuição das Respostas Q21 por Grupo Religioso (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Resposta Q21', fontsize=10)
        ax2.set_ylabel('Percentual (%)', fontsize=10)
        ax2.legend(title='Grupo Religioso')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
    
    # Gráfico 3: Comparação de médias Q20
    if q20_analysis:
        ax3 = axes[1, 0]
        df_q20 = df[[q20_var, 'religiao_grupo']].dropna()
        df_q20.boxplot(column=q20_var, by='religiao_grupo', ax=ax3)
        ax3.set_title('Distribuição das Respostas Q20 por Grupo Religioso', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Grupo Religioso', fontsize=10)
        ax3.set_ylabel('Resposta Q20', fontsize=10)
        ax3.get_figure().suptitle('')  # Remove título automático
    
    # Gráfico 4: Comparação de médias Q21
    if q21_analysis:
        ax4 = axes[1, 1]
        df_q21 = df[[q21_var, 'religiao_grupo']].dropna()
        df_q21.boxplot(column=q21_var, by='religiao_grupo', ax=ax4)
        ax4.set_title('Distribuição das Respostas Q21 por Grupo Religioso', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Grupo Religioso', fontsize=10)
        ax4.set_ylabel('Resposta Q21', fontsize=10)
        ax4.get_figure().suptitle('')  # Remove título automático
    
    plt.tight_layout()
    plt.savefig('comparacao_q20_q21_por_religiao.png', dpi=300, bbox_inches='tight')
    print("Gráficos salvos em: comparacao_q20_q21_por_religiao.png")
    plt.close()

def main():
    """Função principal"""
    sav_file_path = "WVS_Wave_7_Brazil_Spss_v5.1.sav"
    
    # Carregar dados
    df = load_wvs_data(sav_file_path)
    
    # Explorar variáveis
    religiao_vars, q20_vars, q21_vars = explore_variables(df)
    
    # Verificar se encontramos as variáveis necessárias
    if not religiao_vars:
        print("\nAVISO: Variável de religião não encontrada automaticamente.")
        print("Por favor, verifique manualmente as variáveis disponíveis.")
        print("\nVariáveis disponíveis (primeiras 50):")
        for i, col in enumerate(df.columns[:50], 1):
            print(f"  {i}. {col}")
        return
    
    if not q20_vars:
        print("\nAVISO: Variável Q20 não encontrada!")
        return
    
    if not q21_vars:
        print("\nAVISO: Variável Q21 não encontrada!")
        return
    
    # Usar primeira variável encontrada de cada tipo
    religiao_var = religiao_vars[0]
    q20_var = q20_vars[0]
    q21_var = q21_vars[0]
    
    print(f"\nUsando variáveis:")
    print(f"  Religião: {religiao_var}")
    print(f"  Q20: {q20_var}")
    print(f"  Q21: {q21_var}")
    
    # Mapear religião
    df = map_religion(df, religiao_var)
    
    # Analisar Q20
    q20_analysis = analyze_question(df, q20_var, "Q20")
    
    # Analisar Q21
    q21_analysis = analyze_question(df, q21_var, "Q21")
    
    # Criar visualizações
    if q20_analysis and q21_analysis:
        create_visualizations(df, q20_var, q21_var, q20_analysis, q21_analysis)
    
    # Salvar resultados em CSV
    print("\n" + "="*60)
    print("SALVANDO RESULTADOS")
    print("="*60)
    
    # Salvar dados com grupos religiosos
    df.to_csv('wvs_wave7_com_grupos_religiosos.csv', index=False)
    print("Dados completos salvos em: wvs_wave7_com_grupos_religiosos.csv")
    
    # Salvar tabelas de contingência
    if q20_analysis:
        q20_analysis['crosstab'].to_csv('q20_crosstab.csv')
        q20_analysis['crosstab_pct'].to_csv('q20_crosstab_percentual.csv')
        print("Tabelas Q20 salvas em: q20_crosstab.csv e q20_crosstab_percentual.csv")
    
    if q21_analysis:
        q21_analysis['crosstab'].to_csv('q21_crosstab.csv')
        q21_analysis['crosstab_pct'].to_csv('q21_crosstab_percentual.csv')
        print("Tabelas Q21 salvas em: q21_crosstab.csv e q21_crosstab_percentual.csv")
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA!")
    print("="*60)

if __name__ == "__main__":
    main()










