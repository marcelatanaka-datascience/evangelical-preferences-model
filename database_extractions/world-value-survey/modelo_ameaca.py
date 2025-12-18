"""
Script para testar o modelo com efeito da percepção de AMEAÇA INSTITUCIONAL
segundo a teoria proposta na tese.

Modelo teórico:
Yi = β₀ + (ModeloBase) + β₆Ameaçai + γXi + εi

onde:
- Yi: preferência política (y_moral_inv ou y_intitucional_4vars)
- ModeloBase inclui:
  - Freqi: frequência a serviços religiosos (freq_inv_z)
  - Perti: pertencimento a organização religiosa (pert_r_z)
  - Importi: importância da religião na vida (import_relig_z)
  - Identi: autoidentificação como pessoa religiosa (relig_r)
  - GrupoEvi: variável binária evangélico (GrupoEv)
  - Xi: controles sociodemográficos (sexo, raça, escolaridade)
- Ameaçai: percepção de ameaça institucional (ameaca_tipo_inst)

Estimação: OLS com erros robustos de Huber-White (HC3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / 'base_dados_final_consolidada.csv'

# Mapeamento de nomes de variáveis para nomes legíveis
MAPEAMENTO_NOMES = {
    'freq_inv_z': 'Frequência a serviços religiosos',
    'pert_r_z': 'Pertencimento a org. religiosa',
    'import_relig_z': 'Importância da religião',
    'relig_r': 'Autoidentificação religiosa (1=Religioso)',
    'GrupoEv': 'Evangélico',
    'ameaca_tipo_inst': 'Ameaça institucional',
    'dummy_sex': 'Sexo (1=Masculino)',
    'dummy_ethnic': 'Raça/Etnia (1=Branco)',
    'escol_z': 'Escolaridade',
    'const': 'Constante'
}

def preparar_dados(df):
    """
    Prepara os dados para análise, removendo missing e garantindo tipos numéricos.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame com os dados brutos
    
    Returns:
    --------
    DataFrame limpo e preparado
    """
    # Variáveis do modelo
    vars_modelo = [
        'freq_inv_z',      # Freqi
        'pert_r_z',       # Perti
        'import_relig_z', # Importi
        'relig_r',        # Identi
        'GrupoEv',        # GrupoEvi
        'ameaca_tipo_inst',  # Ameaçai
        'dummy_sex',      # Controle: sexo
        'dummy_ethnic',   # Controle: raça/etnia
        'escol_z'         # Controle: escolaridade
    ]
    
    # Variáveis dependentes
    vars_dependentes = ['y_moral_inv', 'y_intitucional_4vars']
    
    # Selecionar apenas variáveis necessárias
    todas_vars = vars_dependentes + vars_modelo
    df_clean = df[todas_vars].copy()
    
    # Garantir que todas as variáveis são numéricas
    for var in todas_vars:
        if not pd.api.types.is_numeric_dtype(df_clean[var]):
            df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')
    
    # Remover missing
    n_antes = len(df_clean)
    df_clean = df_clean.dropna()
    n_depois = len(df_clean)
    
    print(f"  Registros antes: {n_antes:,}")
    print(f"  Registros após remover missing: {n_depois:,}")
    print(f"  Registros removidos: {n_antes - n_depois:,} ({(n_antes - n_depois)/n_antes*100:.2f}%)")
    
    return df_clean

def estimar_modelo(y, X, var_dependente, usar_robustos=True):
    """
    Estima o modelo OLS com erros robustos.
    
    Parameters:
    -----------
    y : array-like
        Variável dependente
    X : array-like ou DataFrame
        Variáveis independentes (sem constante)
    var_dependente : str
        Nome da variável dependente
    usar_robustos : bool
        Se True, usa erros robustos de Huber-White (HC3)
    
    Returns:
    --------
    Objeto do modelo ajustado
    """
    # Adicionar constante
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # Ajustar modelo OLS
    model = OLS(y, X_with_const).fit()
    
    # Re-estimar com erros robustos se solicitado
    if usar_robustos:
        model = model.get_robustcov_results(cov_type='HC3')
    
    return model

def formatar_resultados(model, var_dependente, nomes_variaveis):
    """
    Formata os resultados do modelo em um DataFrame legível.
    
    Parameters:
    -----------
    model : objeto do modelo ajustado
        Modelo OLS ajustado
    var_dependente : str
        Nome da variável dependente
    nomes_variaveis : list
        Lista de nomes das variáveis independentes (sem constante)
    
    Returns:
    --------
    DataFrame com resultados formatados
    """
    # Adicionar constante à lista de variáveis
    todas_vars = ['const'] + nomes_variaveis
    
    # Extrair resultados
    resultados = []
    for i, var in enumerate(todas_vars):
        nome_legivel = MAPEAMENTO_NOMES.get(var, var)
        
        coef = model.params[i]
        std_err = model.bse[i]
        t_stat = model.tvalues[i]
        p_value = model.pvalues[i]
        
        # Intervalo de confiança 95%
        ci = model.conf_int()
        if isinstance(ci, pd.DataFrame):
            ci_lower = ci.iloc[i, 0]
            ci_upper = ci.iloc[i, 1]
        else:
            ci_lower = ci[i, 0]
            ci_upper = ci[i, 1]
        
        # Significância
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        elif p_value < 0.1:
            sig = '.'
        else:
            sig = ''
        
        resultados.append({
            'Variável_Dependente': var_dependente,
            'Variável': var,
            'Nome_Legível': nome_legivel,
            'Coeficiente': coef,
            'Erro_Padrão': std_err,
            't_estatística': t_stat,
            'p_valor': p_value,
            'IC_95_inf': ci_lower,
            'IC_95_sup': ci_upper,
            'Significância': sig
        })
    
    return pd.DataFrame(resultados)

def estatisticas_modelo(model, var_dependente):
    """
    Extrai estatísticas do modelo.
    
    Parameters:
    -----------
    model : objeto do modelo ajustado
        Modelo OLS ajustado
    var_dependente : str
        Nome da variável dependente
    
    Returns:
    --------
    dict com estatísticas do modelo
    """
    return {
        'variável_dependente': var_dependente,
        'n_observações': int(model.nobs),
        'r_quadrado': model.rsquared,
        'r_quadrado_ajustado': model.rsquared_adj,
        'f_estatística': model.fvalue,
        'f_pvalor': model.f_pvalue,
        'log_likelihood': model.llf if hasattr(model, 'llf') else np.nan,
        'aic': model.aic if hasattr(model, 'aic') else np.nan,
        'bic': model.bic if hasattr(model, 'bic') else np.nan
    }

def criar_grafico_coeficientes(resultados_df, output_path):
    """
    Cria um gráfico de coeficientes (forest plot) mostrando os coeficientes
    estimados com seus intervalos de confiança de 95%.
    
    Parameters:
    -----------
    resultados_df : DataFrame
        DataFrame com resultados dos modelos
    output_path : Path
        Caminho para salvar o gráfico
    """
    # Filtrar apenas variáveis independentes (excluir constante)
    df_plot = resultados_df[resultados_df['Variável'] != 'const'].copy()
    
    # Ordenar variáveis (manter ordem lógica)
    ordem_variaveis = [
        'Frequência a serviços religiosos',
        'Pertencimento a org. religiosa',
        'Importância da religião',
        'Autoidentificação religiosa (1=Religioso)',
        'Evangélico',
        'Ameaça institucional',
        'Sexo (1=Masculino)',
        'Raça/Etnia (1=Branco)',
        'Escolaridade'
    ]
    
    # Criar ordem personalizada
    df_plot['Ordem'] = df_plot['Nome_Legível'].apply(
        lambda x: ordem_variaveis.index(x) if x in ordem_variaveis else 999
    )
    df_plot = df_plot.sort_values(['Variável_Dependente', 'Ordem'])
    
    # Preparar dados para o gráfico
    vars_dependentes = df_plot['Variável_Dependente'].unique()
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_alpha(0)  # Fundo transparente
    ax.patch.set_alpha(0)  # Fundo do eixo transparente
    
    # Cores e símbolos para cada variável dependente
    cores_simbolos = {
        'y_moral_inv': {'cor': '#DE1A58', 'marcador': 'o', 'label': 'Preferências Morais'},
        'y_intitucional_4vars': {'cor': '#360185', 'marcador': 's', 'label': 'Preferências Institucionais'}
    }
    
    # Posições no eixo Y
    y_positions = {}
    y_pos = 0
    
    # Criar lista de variáveis únicas (sem duplicatas)
    variaveis_unicas = df_plot['Nome_Legível'].unique()
    
    for var in ordem_variaveis:
        if var in variaveis_unicas:
            y_positions[var] = y_pos
            y_pos += 1
    
    # Plotar coeficientes
    for var_dep in vars_dependentes:
        df_var = df_plot[df_plot['Variável_Dependente'] == var_dep]
        estilo = cores_simbolos[var_dep]
        
        y_coords = []
        x_coords = []
        x_err_lower = []
        x_err_upper = []
        significativos = []
        
        for _, row in df_var.iterrows():
            var_nome = row['Nome_Legível']
            if var_nome in y_positions:
                y_coords.append(y_positions[var_nome])
                x_coords.append(row['Coeficiente'])
                x_err_lower.append(row['Coeficiente'] - row['IC_95_inf'])
                x_err_upper.append(row['IC_95_sup'] - row['Coeficiente'])
                significativos.append(row['p_valor'] < 0.05)
        
        # Plotar intervalos de confiança e coeficientes
        for i, (y, x, err_l, err_u, sig) in enumerate(zip(y_coords, x_coords, x_err_lower, x_err_upper, significativos)):
            # Linha horizontal do intervalo de confiança
            ax.plot([x - err_l, x + err_u], [y, y], 
                   color=estilo['cor'], linewidth=2.5, alpha=0.8, zorder=1)
            
            # Marcador do coeficiente
            if sig:
                # Significativo: marcador preenchido maior e mais escuro
                ax.scatter(x, y, s=180, marker=estilo['marcador'], 
                          color=estilo['cor'], edgecolors='white', 
                          linewidths=2, zorder=3, alpha=1.0, label='_nolegend_')
            else:
                # Não significativo: marcador vazio menor
                ax.scatter(x, y, s=120, marker=estilo['marcador'], 
                          facecolors='none', edgecolors=estilo['cor'], 
                          linewidths=2, zorder=2, alpha=0.8, label='_nolegend_')
    
    # Linha vertical em zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
    
    # Configurar eixos
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=11)
    ax.invert_yaxis()  # Primeira variável no topo
    
    # Título e labels
    ax.set_xlabel('Coeficiente (com IC 95%)', fontsize=13, fontweight='bold')
    ax.set_title('Modelo com Efeito da Ameaça Institucional', 
                 fontsize=15, fontweight='bold', pad=25)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Ajustar limites do eixo X para melhor visualização
    x_min = min([row['IC_95_inf'] for _, row in df_plot.iterrows()])
    x_max = max([row['IC_95_sup'] for _, row in df_plot.iterrows()])
    x_margin = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    
    # Legenda
    legend_elements = []
    for var_dep in vars_dependentes:
        estilo = cores_simbolos[var_dep]
        # Significativo (preenchido)
        legend_elements.append(
            plt.Line2D([0], [0], marker=estilo['marcador'], color='w', 
                      markerfacecolor=estilo['cor'], markersize=12,
                      markeredgecolor='white', markeredgewidth=2,
                      label=f"{estilo['label']} (p<0.05)")
        )
        # Não significativo (vazio)
        legend_elements.append(
            plt.Line2D([0], [0], marker=estilo['marcador'], color='w', 
                      markerfacecolor='none', markersize=10,
                      markeredgecolor=estilo['cor'], markeredgewidth=2,
                      label=f"{estilo['label']} (n.s.)")
        )
    
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, 
             fontsize=11, title='Legenda', title_fontsize=12, 
             edgecolor='gray', fancybox=True)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Gráfico de coeficientes salvo em: {output_path}")

def main():
    """Função principal."""
    print("="*70)
    print("MODELO COM EFEITO DA AMEAÇA INSTITUCIONAL")
    print("="*70)
    print("\nModelo teórico:")
    print("Yi = β₀ + (ModeloBase) + β₆Ameaçai + γXi + εi")
    print("\nEstimação: OLS com erros robustos de Huber-White (HC3)")
    print("="*70)
    
    # Carregar dados
    print(f"\n[1] Carregando base de dados...")
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
        print(f"  ✓ Base carregada: {len(df)} registros, {len(df.columns)} colunas")
    except Exception as e:
        print(f"  ❌ Erro ao carregar base: {e}")
        return
    
    # Preparar dados
    print(f"\n[2] Preparando dados...")
    df_clean = preparar_dados(df)
    
    if len(df_clean) == 0:
        print("  ❌ Nenhum registro válido após limpeza!")
        return
    
    # Definir variáveis independentes
    vars_independentes = [
        'freq_inv_z',      # β₁: Freqi
        'pert_r_z',       # β₂: Perti
        'import_relig_z', # β₃: Importi
        'relig_r',        # β₄: Identi
        'GrupoEv',        # β₅: GrupoEvi
        'ameaca_tipo_inst',  # β₆: Ameaçai
        'dummy_sex',      # γ₁: Sexo
        'dummy_ethnic',   # γ₂: Raça/Etnia
        'escol_z'         # γ₃: Escolaridade
    ]
    
    # Verificar se todas as variáveis existem
    vars_faltantes = [v for v in vars_independentes if v not in df_clean.columns]
    if vars_faltantes:
        print(f"  ❌ Variáveis faltantes: {vars_faltantes}")
        return
    
    # Separar variáveis independentes
    X = df_clean[vars_independentes].values
    
    # Variáveis dependentes
    vars_dependentes = ['y_moral_inv', 'y_intitucional_4vars']
    
    # Estimar modelos
    print(f"\n[3] Estimando modelos...")
    print(f"{'─'*70}")
    
    todos_resultados = []
    todas_estatisticas = []
    
    for var_dep in vars_dependentes:
        if var_dep not in df_clean.columns:
            print(f"\n  ⚠️  {var_dep}: Variável não encontrada, pulando...")
            continue
        
        print(f"\n  [{var_dep}]")
        print(f"  {'─'*60}")
        
        # Separar variável dependente
        y = df_clean[var_dep].values
        
        # Estimar modelo
        model = estimar_modelo(y, X, var_dep, usar_robustos=True)
        
        # Formatar resultados
        resultados_df = formatar_resultados(model, var_dep, vars_independentes)
        todos_resultados.append(resultados_df)
        
        # Estatísticas do modelo
        stats = estatisticas_modelo(model, var_dep)
        todas_estatisticas.append(stats)
        
        # Exibir resultados principais
        print(f"\n  Estatísticas do modelo:")
        print(f"    N observações: {stats['n_observações']:,}")
        print(f"    R²: {stats['r_quadrado']:.4f}")
        print(f"    R² ajustado: {stats['r_quadrado_ajustado']:.4f}")
        print(f"    F-estatística: {stats['f_estatística']:.4f} (p = {stats['f_pvalor']:.6f})")
        
        print(f"\n  Coeficientes:")
        print(f"    {'Variável':<40s} {'Coef.':>10s} {'EP':>10s} {'t':>8s} {'p-valor':>10s} {'Sig.':>5s}")
        print(f"    {'─'*40} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*5}")
        
        for _, row in resultados_df.iterrows():
            print(f"    {row['Nome_Legível']:<40s} {row['Coeficiente']:>10.4f} "
                  f"{row['Erro_Padrão']:>10.4f} {row['t_estatística']:>8.2f} "
                  f"{row['p_valor']:>10.6f} {row['Significância']:>5s}")
        
        print(f"\n  Legenda: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
    
    # Consolidar resultados
    print(f"\n[4] Consolidando resultados...")
    print(f"{'─'*70}")
    
    resultados_completos = pd.concat(todos_resultados, ignore_index=True)
    estatisticas_completas = pd.DataFrame(todas_estatisticas)
    
    # Salvar resultados
    output_coef = BASE_DIR / 'resultados_modelo_ameaca_coeficientes.csv'
    output_stats = BASE_DIR / 'resultados_modelo_ameaca_estatisticas.csv'
    output_grafico = BASE_DIR / 'grafico_coeficientes_modelo_ameaca.png'
    
    resultados_completos.to_csv(output_coef, sep=';', index=False, encoding='utf-8-sig')
    estatisticas_completas.to_csv(output_stats, sep=';', index=False, encoding='utf-8-sig')
    
    print(f"  ✓ Coeficientes salvos em: {output_coef}")
    print(f"  ✓ Estatísticas salvos em: {output_stats}")
    
    # Criar gráfico de coeficientes
    print(f"\n[5] Criando gráfico de coeficientes...")
    print(f"{'─'*70}")
    criar_grafico_coeficientes(resultados_completos, output_grafico)
    
    # Resumo final
    print(f"\n[6] Resumo dos modelos estimados...")
    print(f"{'─'*70}")
    print(estatisticas_completas.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("ESTIMAÇÃO CONCLUÍDA!")
    print("="*70)
    print(f"\nNota: Erros padrão robustos de Huber-White (HC3) foram utilizados")
    print(f"      para corrigir possíveis problemas de heterocedasticidade.")

if __name__ == "__main__":
    main()



