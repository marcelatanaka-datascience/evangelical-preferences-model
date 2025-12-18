#!/usr/bin/env python3
"""
Análise dos respondentes da WVS Wave 7 Brazil separados por religião
Comparação das questões Q20 e Q21 entre evangélicos, católicos e outros

NOTA: Se o arquivo .sav não puder ser lido diretamente, você pode:
1. Abrir o arquivo no SPSS e exportar para CSV
2. Ou usar este script com um arquivo CSV convertido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def try_load_sav(sav_file_path):
    """Tenta carregar arquivo .sav usando pyreadstat"""
    try:
        import pyreadstat
        print(f"Tentando carregar {sav_file_path} com pyreadstat...")
        df, meta = pyreadstat.read_sav(sav_file_path)
        print(f"✓ Arquivo carregado com sucesso! Shape: {df.shape}")
        return df, meta
    except Exception as e:
        print(f"✗ Erro ao carregar com pyreadstat: {e}")
        return None, None

def load_csv(csv_file_path):
    """Carrega arquivo CSV como alternativa"""
    try:
        print(f"Tentando carregar {csv_file_path}...")
        df = pd.read_csv(csv_file_path, encoding='latin1', low_memory=False)
        print(f"✓ Arquivo CSV carregado com sucesso! Shape: {df.shape}")
        return df, None
    except Exception as e:
        print(f"✗ Erro ao carregar CSV: {e}")
        return None, None

def find_variables(df):
    """Encontra as variáveis de religião, Q20 e Q21"""
    print("\n" + "="*60)
    print("PROCURANDO VARIÁVEIS")
    print("="*60)
    
    # Procurar variáveis de religião (WVS geralmente usa Q289, Q289A, ou similar)
    religiao_vars = []
    for col in df.columns:
        col_upper = col.upper()
        if 'Q289' in col_upper or 'RELIG' in col_upper or 'RELIGION' in col_upper:
            religiao_vars.append(col)
    
    # Procurar Q20 e Q21 (exatamente Q20 e Q21, não Q200, Q201, etc.)
    q20_vars = [col for col in df.columns if col.upper() == 'Q20' or col.upper().startswith('Q20_')]
    q21_vars = [col for col in df.columns if col.upper() == 'Q21' or col.upper().startswith('Q21_')]
    
    # Se não encontrou exato, procurar variações
    if not q20_vars:
        q20_vars = [col for col in df.columns if 'Q20' in col.upper() and len(col) <= 5]
    if not q21_vars:
        q21_vars = [col for col in df.columns if 'Q21' in col.upper() and len(col) <= 5]
    
    print(f"Variáveis de religião encontradas: {religiao_vars}")
    print(f"Variáveis Q20 encontradas: {q20_vars}")
    print(f"Variáveis Q21 encontradas: {q21_vars}")
    
    # Se não encontrou, mostrar variáveis que começam com Q
    if not religiao_vars or not q20_vars or not q21_vars:
        print("\nVariáveis que começam com Q (primeiras 30):")
        q_vars = [col for col in df.columns if col.upper().startswith('Q')]
        for var in sorted(q_vars)[:30]:
            print(f"  {var}")
    
    return religiao_vars, q20_vars, q21_vars

def map_religion_wvs(df, religiao_var):
    """Mapeia valores de religião do WVS para categorias"""
    print(f"\n" + "="*60)
    print(f"ANÁLISE DA VARIÁVEL DE RELIGIÃO: {religiao_var}")
    print("="*60)
    
    # Ver valores únicos
    valores_unicos = df[religiao_var].value_counts().sort_index()
    print("\nValores únicos e suas frequências:")
    for valor, count in valores_unicos.items():
        print(f"  {valor}: {count}")
    
    # No WVS, os valores típicos são:
    # 1 = Católico Romano
    # 2 = Protestante
    # 3 = Ortodoxo
    # 4 = Judeu
    # 5 = Islâmico
    # 6 = Hindu
    # 7 = Budista
    # 8 = Outras religiões
    # 9 = Sem religião
    
    def map_religion_value(val):
        if pd.isna(val):
            return 'Outros'
        try:
            val = float(val)
            # Católicos
            if val == 1.0:
                return 'Católicos'
            # Evangélicos/Protestantes
            elif val == 2.0:
                return 'Evangélicos'
            # Outros (incluindo sem religião)
            else:
                return 'Outros'
        except:
            # Se for string, tentar identificar
            val_str = str(val).upper()
            if 'CATÓLIC' in val_str or 'CATHOLIC' in val_str:
                return 'Católicos'
            elif 'EVANGÉLIC' in val_str or 'PROTESTANT' in val_str or 'EVANGELIC' in val_str:
                return 'Evangélicos'
            else:
                return 'Outros'
    
    df['religiao_grupo'] = df[religiao_var].apply(map_religion_value)
    
    print("\nDistribuição por grupo religioso:")
    dist = df['religiao_grupo'].value_counts()
    print(dist)
    print("\nPercentual por grupo religioso:")
    pct = df['religiao_grupo'].value_counts(normalize=True) * 100
    print(pct.round(2))
    
    return df

def analyze_question(df, question_var, question_name):
    """Analisa uma questão específica por grupo religioso"""
    print(f"\n" + "="*60)
    print(f"ANÁLISE DA {question_name}: {question_var}")
    print("="*60)
    
    if question_var not in df.columns:
        print(f"✗ AVISO: Variável {question_var} não encontrada!")
        return None
    
    # Remover valores missing
    df_clean = df[[question_var, 'religiao_grupo']].dropna()
    
    if len(df_clean) == 0:
        print(f"✗ AVISO: Nenhum dado válido encontrado para {question_var}!")
        return None
    
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
    if crosstab_test.shape[0] > 1 and crosstab_test.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(crosstab_test)
        print(f"\nTeste Qui-quadrado:")
        print(f"  Chi-quadrado: {chi2:.4f}")
        print(f"  p-valor: {p_value:.4f}")
        print(f"  Graus de liberdade: {dof}")
        if p_value < 0.05:
            print(f"  ✓ Resultado: Diferenças estatisticamente significativas (p < 0.05)")
        else:
            print(f"  ✗ Resultado: Diferenças não são estatisticamente significativas (p >= 0.05)")
    else:
        chi2, p_value, dof = None, None, None
        print("\nTeste Qui-quadrado não pode ser calculado (poucas categorias)")
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Distribuição Q20 por grupo religioso (barras)
    if q20_analysis and q20_analysis['crosstab_pct'].shape[1] > 0:
        ax1 = axes[0, 0]
        q20_analysis['crosstab_pct'].T.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Distribuição das Respostas Q20 por Grupo Religioso (%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Resposta Q20', fontsize=10)
        ax1.set_ylabel('Percentual (%)', fontsize=10)
        ax1.legend(title='Grupo Religioso', fontsize=9)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Dados Q20 não disponíveis', ha='center', va='center')
        axes[0, 0].set_title('Q20 - Dados não disponíveis')
    
    # Gráfico 2: Distribuição Q21 por grupo religioso (barras)
    if q21_analysis and q21_analysis['crosstab_pct'].shape[1] > 0:
        ax2 = axes[0, 1]
        q21_analysis['crosstab_pct'].T.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Distribuição das Respostas Q21 por Grupo Religioso (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Resposta Q21', fontsize=10)
        ax2.set_ylabel('Percentual (%)', fontsize=10)
        ax2.legend(title='Grupo Religioso', fontsize=9)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Dados Q21 não disponíveis', ha='center', va='center')
        axes[0, 1].set_title('Q21 - Dados não disponíveis')
    
    # Gráfico 3: Comparação de médias Q20
    if q20_analysis:
        ax3 = axes[1, 0]
        df_q20 = df[[q20_var, 'religiao_grupo']].dropna()
        if len(df_q20) > 0:
            df_q20.boxplot(column=q20_var, by='religiao_grupo', ax=ax3)
            ax3.set_title('Distribuição das Respostas Q20 por Grupo Religioso', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Grupo Religioso', fontsize=10)
            ax3.set_ylabel('Resposta Q20', fontsize=10)
            ax3.get_figure().suptitle('')  # Remove título automático
        else:
            ax3.text(0.5, 0.5, 'Dados Q20 não disponíveis', ha='center', va='center')
    
    # Gráfico 4: Comparação de médias Q21
    if q21_analysis:
        ax4 = axes[1, 1]
        df_q21 = df[[q21_var, 'religiao_grupo']].dropna()
        if len(df_q21) > 0:
            df_q21.boxplot(column=q21_var, by='religiao_grupo', ax=ax4)
            ax4.set_title('Distribuição das Respostas Q21 por Grupo Religioso', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Grupo Religioso', fontsize=10)
            ax4.set_ylabel('Resposta Q21', fontsize=10)
            ax4.get_figure().suptitle('')  # Remove título automático
        else:
            ax4.text(0.5, 0.5, 'Dados Q21 não disponíveis', ha='center', va='center')
    
    plt.tight_layout()
    output_file = 'comparacao_q20_q21_por_religiao.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficos salvos em: {output_file}")
    plt.close()

def main():
    """Função principal"""
    sav_file_path = "WVS_Wave_7_Brazil_Spss_v5.1.sav"
    csv_file_path = "WVS_Wave_7_Brazil_Spss_v5.1.csv"
    
    # Tentar carregar .sav
    df, meta = try_load_sav(sav_file_path)
    
    # Se não conseguiu, tentar CSV
    if df is None:
        if os.path.exists(csv_file_path):
            df, meta = load_csv(csv_file_path)
        else:
            print(f"\n✗ Não foi possível carregar o arquivo .sav e o arquivo CSV não foi encontrado.")
            print(f"\nINSTRUÇÕES:")
            print(f"1. Abra o arquivo {sav_file_path} no SPSS")
            print(f"2. Exporte os dados para CSV (File > Save As > CSV)")
            print(f"3. Salve como {csv_file_path} na mesma pasta")
            print(f"4. Execute este script novamente")
            return
    
    if df is None:
        print("✗ Não foi possível carregar os dados.")
        return
    
    # Explorar variáveis
    religiao_vars, q20_vars, q21_vars = find_variables(df)
    
    # Verificar se encontramos as variáveis necessárias
    if not religiao_vars:
        print("\n✗ AVISO: Variável de religião não encontrada automaticamente.")
        print("Por favor, verifique manualmente as variáveis disponíveis.")
        print("\nTodas as variáveis disponíveis:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        return
    
    if not q20_vars:
        print("\n✗ AVISO: Variável Q20 não encontrada!")
        print("Variáveis disponíveis que começam com Q:")
        q_vars = [col for col in df.columns if col.upper().startswith('Q')]
        for var in sorted(q_vars)[:30]:
            print(f"  {var}")
        return
    
    if not q21_vars:
        print("\n✗ AVISO: Variável Q21 não encontrada!")
        print("Variáveis disponíveis que começam com Q:")
        q_vars = [col for col in df.columns if col.upper().startswith('Q')]
        for var in sorted(q_vars)[:30]:
            print(f"  {var}")
        return
    
    # Usar primeira variável encontrada de cada tipo
    religiao_var = religiao_vars[0]
    q20_var = q20_vars[0]
    q21_var = q21_vars[0]
    
    print(f"\n✓ Usando variáveis:")
    print(f"  Religião: {religiao_var}")
    print(f"  Q20: {q20_var}")
    print(f"  Q21: {q21_var}")
    
    # Mapear religião
    df = map_religion_wvs(df, religiao_var)
    
    # Analisar Q20
    q20_analysis = analyze_question(df, q20_var, "Q20")
    
    # Analisar Q21
    q21_analysis = analyze_question(df, q21_var, "Q21")
    
    # Criar visualizações
    if q20_analysis or q21_analysis:
        create_visualizations(df, q20_var, q21_var, q20_analysis, q21_analysis)
    
    # Salvar resultados em CSV
    print("\n" + "="*60)
    print("SALVANDO RESULTADOS")
    print("="*60)
    
    # Salvar dados com grupos religiosos
    output_csv = 'wvs_wave7_com_grupos_religiosos.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✓ Dados completos salvos em: {output_csv}")
    
    # Salvar tabelas de contingência
    if q20_analysis:
        q20_analysis['crosstab'].to_csv('q20_crosstab.csv', encoding='utf-8')
        q20_analysis['crosstab_pct'].to_csv('q20_crosstab_percentual.csv', encoding='utf-8')
        print("✓ Tabelas Q20 salvas")
    
    if q21_analysis:
        q21_analysis['crosstab'].to_csv('q21_crosstab.csv', encoding='utf-8')
        q21_analysis['crosstab_pct'].to_csv('q21_crosstab_percentual.csv', encoding='utf-8')
        print("✓ Tabelas Q21 salvas")
    
    print("\n" + "="*60)
    print("✓ ANÁLISE CONCLUÍDA!")
    print("="*60)

if __name__ == "__main__":
    main()










