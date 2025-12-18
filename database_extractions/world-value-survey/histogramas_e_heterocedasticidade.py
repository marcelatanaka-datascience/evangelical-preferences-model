"""
Script para:
1. Criar histogramas das variáveis dependentes y_moral_inv e y_intitucional_4vars
2. Executar teste de heterocedasticidade (Breusch-Pagan)
3. Executar teste RESET de Ramsey (especificação do modelo)

Teste de Breusch-Pagan:
- H0: Homocedasticidade (variância constante dos resíduos)
- H1: Heterocedasticidade (variância não constante dos resíduos)
- Se p-valor < 0.05, rejeita H0 (há heterocedasticidade)

Teste RESET de Ramsey:
- H0: Modelo está corretamente especificado
- H1: Modelo está incorretamente especificado (formas funcionais incorretas ou variáveis omitidas)
- Se p-valor < 0.05, rejeita H0 (modelo mal especificado)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.regression.linear_model import OLS
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / 'base_dados_final_consolidada.csv'

# Configuração de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')  # Usar whitegrid para melhor contraste com fundo transparente
sns.set_palette("husl")

def criar_histogramas(df, var_name, output_path):
    """
    Cria histograma para uma variável.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame com os dados
    var_name : str
        Nome da variável
    output_path : Path
        Caminho para salvar o gráfico
    """
    # Mapeamento de nomes de variáveis para nomes legíveis
    nomes_legiveis = {
        'y_moral_inv': 'Índice de Preferências Políticas Morais (Y_moral)',
        'y_intitucional_4vars': 'Índice de Preferências Políticas Institucionais (Y_institucional)'
    }
    
    # Obter nome legível ou usar o nome original
    nome_legivel = nomes_legiveis.get(var_name, var_name)
    
    # Filtrar valores válidos
    data = df[var_name].dropna()
    
    if len(data) == 0:
        print(f"  ⚠️  {var_name}: Nenhum valor válido encontrado")
        return
    
    # Criar figura com fundo transparente
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)  # Fundo transparente
    ax.patch.set_alpha(0)  # Fundo do eixo transparente
    
    # Cor 
    cor_verde_pastel = '#DE1A58'  
    
    # Criar histograma com cor verde pastel
    n, bins, patches = ax.hist(data, bins=50, edgecolor='#DE1A58', alpha=0.7, 
                               color=cor_verde_pastel, linewidth=0.5)
    
    # Adicionar linha de densidade (KDE)
    from scipy import stats
    if len(data) > 1:
        x = np.linspace(data.min(), data.max(), 100)
        try:
            kde = stats.gaussian_kde(data)
            y = kde(x) * len(data) * (bins[1] - bins[0])  # Escalar para o histograma
            ax.plot(x, y, '#360185', linewidth=2, label='Curva de densidade (KDE)')
        except:
            pass
    
    # Estatísticas
    mean = data.mean()
    median = data.median()
    std = data.std()
    
    # Adicionar linhas verticais para média e mediana
    ax.axvline(mean, color='#8F0177', linestyle='--', linewidth=2, label=f'Média = {mean:.3f}')
    ax.axvline(median, color='#F4B342', linestyle='--', linewidth=2, label=f'Mediana = {median:.3f}')
    
    # Configurar título e labels
    ax.set_xlabel(nome_legivel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
    ax.set_title(f'Histograma - {nome_legivel}\n(n = {len(data):,}, média = {mean:.3f}, DP = {std:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Adicionar estatísticas no gráfico
    textstr = f'N = {len(data):,}\nMédia = {mean:.3f}\nMediana = {median:.3f}\nDP = {std:.3f}\nMín = {data.min():.3f}\nMáx = {data.max():.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Salvar gráfico com fundo transparente
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Histograma salvo: {output_path}")

def teste_breusch_pagan(y, X, var_name):
    """
    Executa teste de Breusch-Pagan para heterocedasticidade.
    
    Parameters:
    -----------
    y : array-like
        Variável dependente
    X : array-like ou DataFrame
        Variáveis independentes
    var_name : str
        Nome da variável dependente (para exibição)
    
    Returns:
    --------
    dict com resultados do teste
    """
    try:
        # Ajustar modelo OLS
        model = OLS(y, X).fit()
        
        # Executar teste de Breusch-Pagan
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
        
        return {
            'variável_dependente': var_name,
            'lm_statistic': lm,
            'lm_pvalue': lm_pvalue,
            'f_statistic': fvalue,
            'f_pvalue': f_pvalue,
            'heterocedasticidade': 'Sim' if lm_pvalue < 0.05 else 'Não',
            'interpretação': 'Rejeita H0 (heterocedasticidade presente)' if lm_pvalue < 0.05 else 'Não rejeita H0 (homocedasticidade)'
        }
    except Exception as e:
        print(f"  ❌ Erro no teste de Breusch-Pagan para {var_name}: {e}")
        return None

def teste_reset_ramsey(y, X, var_name, power=3):
    """
    Executa teste RESET de Ramsey para especificação do modelo.
    
    Parameters:
    -----------
    y : array-like
        Variável dependente
    X : array-like ou DataFrame
        Variáveis independentes
    var_name : str
        Nome da variável dependente (para exibição)
    power : int
        Potência máxima dos valores ajustados a incluir (padrão: 3)
    
    Returns:
    --------
    dict com resultados do teste
    """
    try:
        # Ajustar modelo OLS
        model = OLS(y, X).fit()
        
        # Executar teste RESET de Ramsey
        reset_result = linear_reset(model, power=power, use_f=True)
        
        # Obter graus de liberdade do modelo original
        df_resid = model.df_resid
        
        return {
            'variável_dependente': var_name,
            'reset_f_statistic': reset_result.fvalue,
            'reset_f_pvalue': reset_result.pvalue,
            'reset_df': df_resid,
            'especificação_correta': 'Não' if reset_result.pvalue < 0.05 else 'Sim',
            'interpretação': 'Rejeita H0 (modelo mal especificado)' if reset_result.pvalue < 0.05 else 'Não rejeita H0 (modelo bem especificado)'
        }
    except Exception as e:
        print(f"  ❌ Erro no teste RESET de Ramsey para {var_name}: {e}")
        return None

def main():
    """Função principal."""
    print("="*70)
    print("HISTOGRAMAS E TESTE DE HETEROCEDASTICIDADE")
    print("="*70)
    
    # Carregar base consolidada
    print(f"\n[1] Carregando base consolidada...")
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
        print(f"  ✓ Base carregada: {len(df)} registros, {len(df.columns)} colunas")
    except Exception as e:
        print(f"  ❌ Erro ao carregar base: {e}")
        return
    
    # Verificar variáveis dependentes
    print(f"\n[2] Verificando variáveis dependentes...")
    vars_dependentes = ['y_moral_inv', 'y_intitucional_4vars']
    
    for var in vars_dependentes:
        if var in df.columns:
            missing = df[var].isna().sum()
            valid = len(df) - missing
            pct = (valid / len(df)) * 100
            print(f"  ✓ {var:25s}: {valid:5d} válidos ({pct:5.2f}%)")
        else:
            print(f"  ❌ {var:25s}: NÃO ENCONTRADA")
    
    # Criar histogramas
    print(f"\n[3] Criando histogramas...")
    print(f"{'─'*70}")
    
    for var in vars_dependentes:
        if var in df.columns:
            output_path = BASE_DIR / f'histograma_{var}.png'
            criar_histogramas(df, var, output_path)
        else:
            print(f"  ⚠️  {var}: Variável não encontrada, pulando histograma")
    
    # Preparar variáveis independentes para o teste de heterocedasticidade
    print(f"\n[4] Preparando variáveis independentes para teste de heterocedasticidade...")
    print(f"{'─'*70}")
    
    # Variáveis independentes (removendo ameaca_tipo_moral e ameaca_tipo_inst devido à multicolinearidade alta)
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
        # Removido: 'ameaca_tipo_moral' (multicolinearidade alta)
        # Removido: 'ameaca_tipo_inst' (multicolinearidade alta)
    ]
    
    # Verificar quais variáveis existem
    vars_independentes_disponiveis = []
    for var in vars_independentes:
        if var in df.columns:
            vars_independentes_disponiveis.append(var)
            missing = df[var].isna().sum()
            print(f"  ✓ {var:20s}: {len(df) - missing:5d} válidos")
        else:
            print(f"  ❌ {var:20s}: NÃO ENCONTRADA")
    
    if not vars_independentes_disponiveis:
        print(f"\n❌ Nenhuma variável independente disponível para o teste!")
        return
    
    # Executar testes de diagnóstico para cada variável dependente
    print(f"\n[5] Executando testes de diagnóstico...")
    print(f"{'─'*70}")
    
    resultados_bp = []
    resultados_reset = []
    
    for var_dep in vars_dependentes:
        if var_dep not in df.columns:
            print(f"\n  ⚠️  {var_dep}: Variável não encontrada, pulando testes")
            continue
        
        print(f"\n  Testando {var_dep}")
        print(f"  {'─'*60}")
        
        # Preparar dados: remover missing
        df_clean = df[[var_dep] + vars_independentes_disponiveis].copy()
        
        # Garantir que todas as variáveis são numéricas
        for var in [var_dep] + vars_independentes_disponiveis:
            if not pd.api.types.is_numeric_dtype(df_clean[var]):
                df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')
        
        # Remover missing
        df_clean = df_clean.dropna()
        
        if len(df_clean) < len(vars_independentes_disponiveis) + 1:
            print(f"  ⚠️  Dados insuficientes: {len(df_clean)} registros válidos")
            continue
        
        # Separar variáveis
        y = df_clean[var_dep].values
        X = df_clean[vars_independentes_disponiveis].values
        
        # Adicionar constante
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Executar teste de Breusch-Pagan
        print(f"\n  [5.1] Teste de Breusch-Pagan (Heterocedasticidade):")
        resultado_bp = teste_breusch_pagan(y, X_with_const, var_dep)
        
        if resultado_bp:
            resultados_bp.append(resultado_bp)
            
            print(f"    Estatística LM:     {resultado_bp['lm_statistic']:.4f}")
            print(f"    p-valor (LM):       {resultado_bp['lm_pvalue']:.6f}")
            print(f"    Estatística F:      {resultado_bp['f_statistic']:.4f}")
            print(f"    p-valor (F):        {resultado_bp['f_pvalue']:.6f}")
            print(f"    Heterocedasticidade: {resultado_bp['heterocedasticidade']}")
            print(f"    Interpretação:      {resultado_bp['interpretação']}")
            
            # Interpretação detalhada
            if resultado_bp['lm_pvalue'] < 0.05:
                print(f"    ⚠️  ATENÇÃO: Evidência de heterocedasticidade (p < 0.05)")
                print(f"       Isso pode afetar a eficiência dos estimadores e a validade dos testes.")
            else:
                print(f"    ✓ Não há evidência de heterocedasticidade (p >= 0.05)")
                print(f"       A suposição de homocedasticidade parece ser satisfeita.")
        
        # Executar teste RESET de Ramsey
        print(f"\n  [5.2] Teste RESET de Ramsey (Especificação do Modelo):")
        resultado_reset = teste_reset_ramsey(y, X_with_const, var_dep)
        
        if resultado_reset:
            resultados_reset.append(resultado_reset)
            
            print(f"    Estatística F:      {resultado_reset['reset_f_statistic']:.4f}")
            print(f"    p-valor (F):        {resultado_reset['reset_f_pvalue']:.6f}")
            print(f"    Graus de liberdade: {resultado_reset['reset_df']:.0f}")
            print(f"    Especificação:      {resultado_reset['especificação_correta']}")
            print(f"    Interpretação:      {resultado_reset['interpretação']}")
            
            # Interpretação detalhada
            if resultado_reset['reset_f_pvalue'] < 0.05:
                print(f"    ⚠️  ATENÇÃO: Evidência de má especificação do modelo (p < 0.05)")
                print(f"       Pode haver formas funcionais incorretas ou variáveis omitidas.")
            else:
                print(f"    ✓ Não há evidência de má especificação (p >= 0.05)")
                print(f"       O modelo parece estar corretamente especificado.")
    
    # Mapeamento de nomes de variáveis para nomes legíveis
    mapeamento_nomes = {
        'freq_inv_z': 'Frequência a serviços religiosos',
        'pert_r_z': 'Pertencimento a org. religiosa',
        'import_relig_z': 'Importância da religião',
        'relig_r': 'Religiosidade Individual',
        'GrupoEv': 'Religião',
        'dummy_ethnic': 'Raça/Etnia',
        'dummy_sex': 'Sexo',
        'escol_z': 'Escolaridade',
        'ano_ord': 'Ano Ordinal',
        'ameaca_tipo_moral': 'Ameaça moral',
        'y_moral_inv': 'Índice de Preferências Políticas Morais',
        'y_intitucional_4vars': 'Índice de Preferências Políticas Institucionais'
    }
    
    # Salvar resultados
    print(f"\n[6] Salvando resultados...")
    print(f"{'─'*70}")
    
    if resultados_bp:
        resultados_bp_df = pd.DataFrame(resultados_bp)
        output_file_bp = BASE_DIR / 'resultados_teste_heterocedasticidade.csv'
        resultados_bp_df.to_csv(output_file_bp, sep=';', index=False, encoding='utf-8-sig')
        print(f"  ✓ Resultados Breusch-Pagan salvos em: {output_file_bp}")
    
    if resultados_reset:
        resultados_reset_df = pd.DataFrame(resultados_reset)
        output_file_reset = BASE_DIR / 'resultados_teste_reset_ramsey.csv'
        resultados_reset_df.to_csv(output_file_reset, sep=';', index=False, encoding='utf-8-sig')
        print(f"  ✓ Resultados RESET de Ramsey salvos em: {output_file_reset}")
    
    # Criar tabela final consolidada com nomes legíveis
    print(f"\n[7] Criando tabela final consolidada...")
    print(f"{'─'*70}")
    
    tabela_final = []
    
    # Adicionar informações das variáveis independentes
    for var in vars_independentes_disponiveis:
        nome_legivel = mapeamento_nomes.get(var, var)
        missing = df[var].isna().sum()
        valid = len(df) - missing
        
        # Estatísticas descritivas
        if valid > 0:
            dados_validos = df[var].dropna()
            media = dados_validos.mean() if pd.api.types.is_numeric_dtype(dados_validos) else np.nan
            desvio = dados_validos.std() if pd.api.types.is_numeric_dtype(dados_validos) else np.nan
        else:
            media = np.nan
            desvio = np.nan
        
        tabela_final.append({
            'Variável Original': var,
            'Nome Legível': nome_legivel,
            'Tipo': 'Independente',
            'N Válidos': valid,
            'N Missing': missing,
            'Média': round(media, 4) if not pd.isna(media) else np.nan,
            'Desvio Padrão': round(desvio, 4) if not pd.isna(desvio) else np.nan
        })
    
    # Adicionar informações das variáveis dependentes
    for var in vars_dependentes:
        if var in df.columns:
            nome_legivel = mapeamento_nomes.get(var, var)
            missing = df[var].isna().sum()
            valid = len(df) - missing
            
            # Estatísticas descritivas
            if valid > 0:
                dados_validos = df[var].dropna()
                media = dados_validos.mean()
                desvio = dados_validos.std()
            else:
                media = np.nan
                desvio = np.nan
            
            tabela_final.append({
                'Variável Original': var,
                'Nome Legível': nome_legivel,
                'Tipo': 'Dependente',
                'N Válidos': valid,
                'N Missing': missing,
                'Média': round(media, 4) if not pd.isna(media) else np.nan,
                'Desvio Padrão': round(desvio, 4) if not pd.isna(desvio) else np.nan
            })
    
    # Adicionar resultados dos testes
    if resultados_bp:
        for resultado in resultados_bp:
            var_dep = resultado['variável_dependente']
            nome_legivel = mapeamento_nomes.get(var_dep, var_dep)
            
            # Encontrar linha correspondente e adicionar resultados
            for item in tabela_final:
                if item['Variável Original'] == var_dep:
                    item['BP_LM_Estatística'] = round(resultado['lm_statistic'], 4)
                    item['BP_LM_pvalor'] = round(resultado['lm_pvalue'], 6)
                    item['BP_Heterocedasticidade'] = resultado['heterocedasticidade']
                    break
    
    if resultados_reset:
        for resultado in resultados_reset:
            var_dep = resultado['variável_dependente']
            
            # Encontrar linha correspondente e adicionar resultados
            for item in tabela_final:
                if item['Variável Original'] == var_dep:
                    item['RESET_F_Estatística'] = round(resultado['reset_f_statistic'], 4)
                    item['RESET_F_pvalor'] = round(resultado['reset_f_pvalue'], 6)
                    item['RESET_Especificação'] = resultado['especificação_correta']
                    break
    
    # Criar DataFrame final
    tabela_final_df = pd.DataFrame(tabela_final)
    
    # Salvar tabela final
    output_tabela = BASE_DIR / 'tabela_final_variaveis_diagnosticos.csv'
    tabela_final_df.to_csv(output_tabela, sep=';', index=False, encoding='utf-8-sig')
    print(f"  ✓ Tabela final consolidada salva em: {output_tabela}")
    
    # Exibir resumo
    print(f"\n[8] Resumo dos resultados...")
    print(f"{'─'*70}")
    
    if resultados_bp:
        print(f"\n  Teste de Breusch-Pagan (Heterocedasticidade):")
        resultados_bp_df = pd.DataFrame(resultados_bp)
        print(resultados_bp_df.to_string(index=False))
    
    if resultados_reset:
        print(f"\n  Teste RESET de Ramsey (Especificação do Modelo):")
        resultados_reset_df = pd.DataFrame(resultados_reset)
        print(resultados_reset_df.to_string(index=False))
    
    print(f"\n  Tabela Final Consolidada (primeiras linhas):")
    print(tabela_final_df.head(10).to_string(index=False))
    
    print(f"\n{'='*70}")
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)
    
    # Recomendações
    print(f"\n[9] Recomendações...")
    print(f"{'─'*70}")
    
    heterocedasticas = [r for r in resultados_bp if r['heterocedasticidade'] == 'Sim']
    mal_especificados = [r for r in resultados_reset if r['especificação_correta'] == 'Não']
    
    if heterocedasticas:
        print(f"  ⚠️  {len(heterocedasticas)} variável(is) dependente(s) apresentam heterocedasticidade:")
        for r in heterocedasticas:
            print(f"     - {r['variável_dependente']}")
        print(f"\n  Recomendações para heterocedasticidade:")
        print(f"     - Usar erros padrão robustos (Huber-White)")
        print(f"     - Considerar transformações nas variáveis (log, raiz quadrada)")
        print(f"     - Usar modelos que lidam com heterocedasticidade (GLS, WLS)")
    else:
        print(f"  ✓ Nenhuma evidência de heterocedasticidade encontrada.")
        print(f"     A suposição de homocedasticidade parece ser satisfeita para todas as variáveis.")
    
    if mal_especificados:
        print(f"\n  ⚠️  {len(mal_especificados)} variável(is) dependente(s) apresentam má especificação:")
        for r in mal_especificados:
            print(f"     - {r['variável_dependente']}")
        print(f"\n  Recomendações para má especificação:")
        print(f"     - Verificar formas funcionais (considerar termos não-lineares)")
        print(f"     - Verificar se há variáveis importantes omitidas")
        print(f"     - Considerar interações entre variáveis")
        print(f"     - Revisar a teoria subjacente ao modelo")
    else:
        print(f"\n  ✓ Nenhuma evidência de má especificação encontrada.")
        print(f"     Os modelos parecem estar corretamente especificados.")

if __name__ == "__main__":
    main()

