"""
Script para criar um índice de religiosidade composto a partir das ondas do World Values Survey.

O índice é composto por 4 dimensões:
1. Saliencia Religiosa
2. Religiosidade Individual
3. Frequência a Serviços Religiosos
4. Pertenimento a Organização Religiosa
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm, multivariate_normal
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression
from numpy.linalg import cond, det

# Configuração dos caminhos dos arquivos
BASE_DIR = Path(__file__).parent

# Mapeamento de variáveis por onda e dimensão
VARIABLES_MAPPING = {
    'wave-2-1991-br': {
        'file': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V151',
        'frequencia': 'V147',
        'organizacao': 'V20'
    },
    'wave-3-1997-br': {
        'file': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V182',
        'frequencia': 'V181',
        'organizacao': 'V28'  # V28 é a variável para igreja/organização religiosa
    },
    'wave-5-2006-br': {
        'file': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V187',
        'frequencia': 'V186',
        'organizacao': 'V24'
    },
    'wave-6-2014-br': {
        'file': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'saliencia': 'V9',
        'religiosidade_individual': 'V147',
        'frequencia': 'V145',
        'organizacao': 'V25'
    },
    'wave-7-2018-br': {
        'file': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'saliencia': 'Q6',
        'religiosidade_individual': 'Q173',
        'frequencia': 'Q171',
        'organizacao': 'Q94'
    }
}


def load_wave_data(wave_dir, file_name, wave_config):
    """
    Carrega os dados de uma onda específica, lendo apenas as colunas necessárias.
    
    Parâmetros:
    -----------
    wave_dir : str
        Diretório da onda
    file_name : str
        Nome do arquivo CSV
    wave_config : dict
        Configuração da onda com as variáveis necessárias
    
    Retorna:
    --------
    DataFrame
        DataFrame com apenas as colunas especificadas
    """
    file_path = BASE_DIR / wave_dir / file_name
    
    if not file_path.exists():
        print(f"Arquivo não encontrado: {file_path}")
        return None
    
    # Define as colunas que precisamos ler
    columns_to_read = [
        wave_config['saliencia'],
        wave_config['religiosidade_individual'],
        wave_config['frequencia'],
        wave_config['organizacao']
    ]
    
    # Adiciona coluna de ID se disponível (V1 para waves 2-6, D_INTERVIEW para wave 7)
    if wave_dir == 'wave-7-2018-br':
        id_column = 'D_INTERVIEW'
    else:
        id_column = 'V1'
    
    if id_column not in columns_to_read:
        columns_to_read.append(id_column)
    
    try:
        # Primeiro, lê apenas o cabeçalho para verificar quais colunas existem
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Remove aspas e divide por ponto-e-vírgula
            header_columns = [col.strip('"') for col in first_line.split(';')]
        
        # Verifica quais colunas realmente existem no arquivo
        available_columns = []
        missing_columns = []
        
        for col in columns_to_read:
            if col in header_columns:
                available_columns.append(col)
            else:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"  Aviso: Colunas não encontradas: {missing_columns}")
        
        if not available_columns:
            print(f"  Erro: Nenhuma das colunas necessárias foi encontrada")
            return None
        
        # Lê o CSV especificando explicitamente as colunas a serem lidas
        df = pd.read_csv(
            file_path,
            sep=';',
            encoding='utf-8',
            usecols=available_columns,
            low_memory=False,
            quotechar='"'
        )
        
        # Remove aspas dos nomes das colunas se existirem
        df.columns = df.columns.str.strip('"')
        
        print(f"  Colunas lidas: {list(df.columns)}")
        
    except Exception as e:
        print(f"Erro ao ler {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return df


def normalize_saliencia(value, wave):
    """
    Normaliza valores de saliência religiosa para escala 0-1.
    Valores mais altos = maior importância da religião.
    """
    if pd.isna(value) or value < 0:
        return np.nan
    
    # Wave 2: 1=Muito importante, 2=Importante, 3=Pouco importante, 4=Nada importante
    if wave == 'wave-2-1991-br':
        if value == 1:
            return 1.0
        elif value == 2:
            return 0.67
        elif value == 3:
            return 0.33
        elif value == 4:
            return 0.0
        else:
            return np.nan
    
    # Waves 3, 5, 6, 7: 1=Muito importante, 2=Importante, 3=Pouco importante, 4=Nada importante
    else:
        if value == 1:
            return 1.0
        elif value == 2:
            return 0.67
        elif value == 3:
            return 0.33
        elif value == 4:
            return 0.0
        else:
            return np.nan


def normalize_religiosidade_individual(value, wave):
    """
    Normaliza valores de religiosidade individual para escala 0-1.
    Valores mais altos = mais religioso.
    """
    if pd.isna(value) or value < 0:
        return np.nan
    
    # Wave 2: 1=Religioso, 2=Não religioso, 3=Ateu
    if wave == 'wave-2-1991-br':
        if value == 1:
            return 1.0
        elif value == 2:
            return 0.0
        elif value == 3:
            return 0.0
        else:
            return np.nan
    
    # Wave 3: 1=Religioso, 2=Não religioso, 3=Ateu
    elif wave == 'wave-3-1997-br':
        if value == 1:
            return 1.0
        elif value == 2:
            return 0.0
        elif value == 3:
            return 0.0
        else:
            return np.nan
    
    # Waves 5, 6, 7: 1=Religioso, 2=Não religioso, 3=Ateu
    else:
        if value == 1:
            return 1.0
        elif value == 2:
            return 0.00
        elif value == 3:
            return 0.0
        else:
            return np.nan


def normalize_frequencia(value, wave):
    """
    Normaliza valores de frequência a serviços religiosos para escala 0-1.
    Valores mais altos = maior frequência.
    """
    if pd.isna(value) or value < 0:
        return np.nan
    
    # Wave 2: V147 - 1=Sim/Frequente, 2=Não/Raro
    if wave == 'wave-2-1991-br':
        if value == 1:  # Mais de uma vez por semana
            return 1.0
        elif value == 2:  # Uma vez por semana
            return 0.85
        elif value == 3:  # Uma vez por mês
            return 0.65
        elif value == 4:  # Natal/Pascoa
            return 0.45
        elif value == 5:  # Festas religiosas
            return 0.25
        elif value == 6:  # Uma vez por ano
            return 0.15
        elif value == 7:  # Mais raramente
            return 0.0
        elif value == 8:  # Nunca/Praticamente nunca
            return 0.0
        else:
            return np.nan
    
    # Wave 3: V181 - 1=Frequente, 2=Ocasional, 3=Raro
    elif wave == 'wave-3-1997-br':
        if value == 1:  # Mais de uma vez por semana
            return 1.0
        elif value == 2:  # Uma vez por semana
            return 0.85
        elif value == 3:  # Uma vez por mês
            return 0.65
        elif value == 4:  # Natal/Pascoa
            return 0.45
        elif value == 5:  # Uma vez por ano
            return 0.25
        elif value == 6:  # Mais Raramente
            return 0.15
        elif value == 7:  # Nunca/Praticamente nunca
            return 0.0
        else:
            return np.nan
    
    # Wave 5: V186 - 1=Frequente, 2=Ocasional, 3=Raro
    elif wave == 'wave-5-2006-br':
        if value == 1:  # Mais de uma vez por semana
            return 1.0
        elif value == 2:  # Uma vez por semana
            return 0.85
        elif value == 3:  # Uma vez por mês
            return 0.65
        elif value == 4:  # Dias santos
            return 0.45
        elif value == 5:  # Uma vez por ano
            return 0.25
        elif value == 6:  # Mais Raramente
            return 0.15
        elif value == 7:  # Nunca/Praticamente nunca
            return 0.0
        else:
            return np.nan
    
    # Wave 6: V145 - Escala mais detalhada (1-8)
    # Wave 7: Q171 - Escala mais detalhada (1-8)
    # 1=Mais de uma vez por semana, 2=Uma vez por semana, 3=Uma vez por mês,
    # 4=Ocasionalmente, 5=Uma vez por ano, 6=Menos frequentemente, 7=Nunca, 8=Não sabe
    else:
        if value == 1:  # Mais de uma vez por semana
            return 1.0
        elif value == 2:  # Uma vez por semana
            return 0.85
        elif value == 3:  # Uma vez por mês
            return 0.65
        elif value == 4:  # Dias santos
            return 0.45
        elif value == 5:  # Uma vez por ano
            return 0.25
        elif value == 6:  # Mais Raramente
            return 0.15
        elif value == 7:  # Nunca/Praticamente nunca
            return 0.0
        else:
            return np.nan


def normalize_organizacao(value, wave):
    """
    Normaliza valores de pertencimento a organização religiosa para escala 0-1.
    Valores mais altos = maior participação.
    """
    if pd.isna(value) or value < 0:
        return np.nan
    
    # Wave 2: V20 - 1=Participa, 2=Não Participa
    if wave == 'wave-2-1991-br':
        if value == 1:
            return 1.0
        elif value == 2:
            return 0.0
        else:
            return np.nan
    
    # Wave 3: V28 - 1=Participa ativamente, 2=Participa sem atuar, 3=Não participa
    elif wave == 'wave-3-1997-br':
        if value == 1:  # Participa ativamente
            return 1.0
        elif value == 2:  # Participa sem atuar
            return 0.5
        elif value == 3:  # Não participa
            return 0.0
        else:
            return np.nan
    
    # Waves 5, 6, 7: 0=Não pertence, 1=Pertence e  e não participa, 2=Pertence e participa
    else:
        if value == 2:  # Pertence e participa
            return 1.0
        elif value == 1:  # Pertence mas não participa
            return 0.5
        elif value == 0:  # Não pertence
            return 0.0
        else:
            return np.nan


def process_wave(wave_name, wave_config):
    """Processa uma onda específica e retorna DataFrame com dimensões normalizadas."""
    print(f"\nProcessando {wave_name}...")
    
    df = load_wave_data(wave_name, wave_config['file'], wave_config)
    if df is None:
        return None
    
    # Adiciona identificador da onda
    df['wave'] = wave_name
    df['year'] = wave_name.split('-')[1] if '-' in wave_name else ''
    
    # Processa cada dimensão
    results = {}
    
    # 1. Saliencia Religiosa
    if wave_config['saliencia'] in df.columns:
        results['saliencia'] = df[wave_config['saliencia']].apply(
            lambda x: normalize_saliencia(x, wave_name)
        )
    else:
        print(f"  Variável {wave_config['saliencia']} não encontrada")
        results['saliencia'] = pd.Series([np.nan] * len(df))
    
    # 2. Religiosidade Individual
    if wave_config['religiosidade_individual'] in df.columns:
        results['religiosidade_individual'] = df[wave_config['religiosidade_individual']].apply(
            lambda x: normalize_religiosidade_individual(x, wave_name)
        )
    else:
        print(f"  Variável {wave_config['religiosidade_individual']} não encontrada")
        results['religiosidade_individual'] = pd.Series([np.nan] * len(df))
    
    # 3. Frequência a Serviços Religiosos
    if wave_config['frequencia'] in df.columns:
        results['frequencia'] = df[wave_config['frequencia']].apply(
            lambda x: normalize_frequencia(x, wave_name)
        )
    else:
        print(f"  Variável {wave_config['frequencia']} não encontrada")
        results['frequencia'] = pd.Series([np.nan] * len(df))
    
    # 4. Pertenimento a Organização Religiosa
    if wave_config['organizacao'] in df.columns:
        results['organizacao'] = df[wave_config['organizacao']].apply(
            lambda x: normalize_organizacao(x, wave_name)
        )
    else:
        print(f"  Variável {wave_config['organizacao']} não encontrada")
        results['organizacao'] = pd.Series([np.nan] * len(df))
    
    # Cria DataFrame com resultados
    result_df = pd.DataFrame({
        'wave': df['wave'],
        'year': df['year'],
        'saliencia': results['saliencia'],
        'religiosidade_individual': results['religiosidade_individual'],
        'frequencia': results['frequencia'],
        'organizacao': results['organizacao']
    })
    
    # Calcula índice composto (média das 4 dimensões)
    result_df['indice_religiosidade'] = result_df[
        ['saliencia', 'religiosidade_individual', 'frequencia', 'organizacao']
    ].mean(axis=1)
    
    # Adiciona informações originais se disponíveis
    if 'V1' in df.columns:
        result_df['id_original'] = df['V1'].values
    elif 'D_INTERVIEW' in df.columns:
        result_df['id_original'] = df['D_INTERVIEW'].values
    
    print(f"  Processados {len(result_df)} registros")
    print(f"  Taxa de completude: {result_df['indice_religiosidade'].notna().sum() / len(result_df) * 100:.1f}%")
    
    return result_df


def cronbach_alpha(df, items):
    """
    Calcula o Alfa de Cronbach para avaliar a consistência interna.
    
    Nota: Para dados categóricos/ordinais, o Alfa de Cronbach pode não ser ideal.
    Considerar também outras medidas de consistência interna.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame com os dados
    items : list
        Lista com os nomes das colunas (itens) a serem avaliados
    
    Retorna:
    --------
    float
        Valor do Alfa de Cronbach (0 a 1, onde valores > 0.7 indicam boa consistência)
    """
    # Remove linhas com valores missing em qualquer item
    df_clean = df[items].dropna()
    
    if len(df_clean) < 2:
        return np.nan
    
    # Calcula variâncias
    item_variances = df_clean[items].var(axis=0, ddof=1)
    total_variance = df_clean[items].sum(axis=1).var(ddof=1)
    
    # Número de itens
    k = len(items)
    
    # Fórmula do Alfa de Cronbach
    if total_variance == 0:
        return np.nan
    
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    
    return alpha


def calculate_cronbach_alpha(combined_df):
    """Calcula o Alfa de Cronbach para cada onda e para o conjunto completo."""
    dimensoes = ['saliencia', 'religiosidade_individual', 'frequencia', 'organizacao']
    
    print("\n" + "=" * 60)
    print("TESTE DE ALFA DE CRONBACH")
    print("=" * 60)
    print("\nO Alfa de Cronbach mede a consistência interna entre as dimensões do índice.")
    print("NOTA: Para dados categóricos/ordinais, o Alfa de Cronbach pode não ser ideal.")
    print("      Considere também as correlações de Spearman e Kendall Tau.")
    print("\nInterpretação do Alfa de Cronbach:")
    print("  α < 0.5:  Consistência inadequada")
    print("  0.5 ≤ α < 0.7: Consistência aceitável")
    print("  0.7 ≤ α < 0.9: Consistência boa")
    print("  α ≥ 0.9: Consistência excelente")
    
    results = []
    
    # Por onda
    print("\n" + "-" * 60)
    print("ALFA DE CRONBACH POR ONDA:")
    print("-" * 60)
    
    for wave in combined_df['wave'].unique():
        df_wave = combined_df[combined_df['wave'] == wave]
        alpha = cronbach_alpha(df_wave, dimensoes)
        
        # Interpretação
        if pd.isna(alpha):
            interp = "Não calculável"
        elif alpha < 0.5:
            interp = "Inadequada"
        elif alpha < 0.7:
            interp = "Aceitável"
        elif alpha < 0.9:
            interp = "Boa"
        else:
            interp = "Excelente"
        
        print(f"{wave:25s}: α = {alpha:.4f} ({interp})")
        results.append({
            'wave': wave,
            'cronbach_alpha': alpha,
            'interpretacao': interp,
            'n': len(df_wave[dimensoes].dropna())
        })
    
    # Conjunto completo
    print("\n" + "-" * 60)
    print("ALFA DE CRONBACH - CONJUNTO COMPLETO:")
    print("-" * 60)
    
    alpha_total = cronbach_alpha(combined_df, dimensoes)
    
    if pd.isna(alpha_total):
        interp_total = "Não calculável"
    elif alpha_total < 0.5:
        interp_total = "Inadequada"
    elif alpha_total < 0.7:
        interp_total = "Aceitável"
    elif alpha_total < 0.9:
        interp_total = "Boa"
    else:
        interp_total = "Excelente"
    
    print(f"Todas as ondas: α = {alpha_total:.4f} ({interp_total})")
    results.append({
        'wave': 'Todas as ondas',
        'cronbach_alpha': alpha_total,
        'interpretacao': interp_total,
        'n': len(combined_df[dimensoes].dropna())
    })
    
    # Análise de correlações entre dimensões
    # Como os dados são categóricos/ordinais, usamos correlações apropriadas
    print("\n" + "-" * 60)
    print("MATRIZES DE CORRELAÇÃO ENTRE DIMENSÕES (Conjunto Completo):")
    print("-" * 60)
    print("\nNota: Como os dados são categóricos/ordinais, calculamos:")
    print("  - Correlação de Pearson (para comparação)")
    print("  - Correlação de Spearman (rank correlation - apropriada para ordinais)")
    print("  - Correlação de Kendall Tau (também apropriada para ordinais)")
    
    # Correlação de Pearson (tradicional, mas pode não ser ideal para categóricos)
    corr_pearson = combined_df[dimensoes].corr(method='pearson')
    print("\n1. CORRELAÇÃO DE PEARSON (r):")
    print(corr_pearson.round(4))
    
    # Correlação de Spearman (rank correlation - melhor para ordinais)
    corr_spearman = combined_df[dimensoes].corr(method='spearman')
    print("\n2. CORRELAÇÃO DE SPEARMAN (ρ - rho):")
    print(corr_spearman.round(4))
    
    # Correlação de Kendall Tau (também para ordinais)
    corr_kendall = combined_df[dimensoes].corr(method='kendall')
    print("\n3. CORRELAÇÃO DE KENDALL TAU (τ - tau):")
    print(corr_kendall.round(4))
    
    # Salva todas as matrizes
    corr_file_pearson = BASE_DIR / 'correlacao_pearson_dimensoes.csv'
    corr_file_spearman = BASE_DIR / 'correlacao_spearman_dimensoes.csv'
    corr_file_kendall = BASE_DIR / 'correlacao_kendall_dimensoes.csv'
    
    corr_pearson.to_csv(corr_file_pearson, encoding='utf-8')
    corr_spearman.to_csv(corr_file_spearman, encoding='utf-8')
    corr_kendall.to_csv(corr_file_kendall, encoding='utf-8')
    
    print(f"\nMatrizes de correlação salvas:")
    print(f"  - Pearson: {corr_file_pearson}")
    print(f"  - Spearman: {corr_file_spearman}")
    print(f"  - Kendall: {corr_file_kendall}")
    
    # Salva resultados
    cronbach_df = pd.DataFrame(results)
    cronbach_file = BASE_DIR / 'cronbach_alpha_indice_religiosidade.csv'
    cronbach_df.to_csv(cronbach_file, index=False, encoding='utf-8')
    print(f"Resultados do Alfa de Cronbach salvos em: {cronbach_file}")
    
    # Nota sobre interpretação
    print("\n" + "-" * 60)
    print("NOTA SOBRE OS RESULTADOS:")
    print("-" * 60)
    print("Valores negativos ou muito baixos de Alfa de Cronbach podem indicar:")
    print("1. As dimensões medem aspectos diferentes da religiosidade")
    print("2. Algumas dimensões podem ter correlações negativas")
    print("3. A religiosidade pode ser um construto multidimensional")
    print("4. Para dados categóricos/ordinais, o Alfa de Cronbach pode não ser")
    print("   a medida mais apropriada - verifique as correlações de Spearman")
    print("   e Kendall Tau acima.")
    print("\nIsso não necessariamente invalida o índice, mas sugere que as")
    print("dimensões capturam aspectos distintos do fenômeno religioso.")
    
    return cronbach_df


def bivariate_normal_cdf(x, y, rho):
    """
    Calcula a CDF da distribuição normal bivariada padrão.
    Usa aproximação baseada em expansão de série.
    """
    if abs(rho) < 0.01:
        return norm.cdf(x) * norm.cdf(y)
    
    try:
        # Usa aproximação baseada em expansão de série
        x_norm = norm.cdf(x)
        y_norm = norm.cdf(y)
        
        if abs(rho) > 0.95:
            # Para correlações muito altas, usa aproximação
            return min(x_norm, y_norm) if rho > 0 else max(0, x_norm + y_norm - 1)
        
        # Aproximação usando expansão de série
        # P(X <= x, Y <= y) ≈ Φ(x)Φ(y) + φ(x)φ(y) * [ρ + ρ²(xy)/2 + ...]
        phi_x = norm.pdf(x) if not (np.isinf(x) or np.isnan(x)) else 0.0
        phi_y = norm.pdf(y) if not (np.isinf(y) or np.isnan(y)) else 0.0
        
        # Primeira ordem
        correction = rho * phi_x * phi_y
        
        # Segunda ordem (aproximação) - apenas se valores são finitos
        if abs(rho) > 0.3 and np.isfinite(x) and np.isfinite(y):
            correction += (rho**2) * phi_x * phi_y * x * y / 2
        
        result = x_norm * y_norm + correction
        return max(0.0, min(1.0, result))
    except:
        # Fallback simples
        try:
            return norm.cdf(x) * norm.cdf(y)
        except:
            return 0.5


def polychoric_correlation(x, y):
    """
    Calcula correlação policórica entre duas variáveis ordinais.
    
    A correlação policórica assume que há variáveis latentes contínuas
    subjacentes às variáveis ordinais observadas.
    
    Implementação baseada em máxima verossimilhança usando otimização.
    
    Parâmetros:
    -----------
    x, y : array-like
        Variáveis ordinais (podem ter diferentes números de categorias)
    
    Retorna:
    --------
    float
        Correlação policórica estimada
    """
    # Remove valores missing
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = np.array(x[mask])
    y_clean = np.array(y[mask])
    
    if len(x_clean) < 10:
        return np.nan
    
    # Converte para inteiros (categorias) começando de 0
    x_cats = pd.Categorical(x_clean).codes
    y_cats = pd.Categorical(y_clean).codes
    
    # Se ambas são dicotômicas (2 categorias), usa correlação tetraicórica
    if len(np.unique(x_cats)) == 2 and len(np.unique(y_cats)) == 2:
        return tetrachoric_correlation(x_cats, y_cats)
    
    # Para variáveis com mais categorias, calcula correlação policórica
    try:
        # Usa correlação de Spearman como aproximação inicial
        rho_init = stats.spearmanr(x_clean, y_clean)[0]
        
        if np.isnan(rho_init):
            return np.nan
        
        # Limita valor inicial
        rho_init = max(-0.95, min(0.95, rho_init))
        
        # Calcula thresholds (pontos de corte) para cada variável
        x_unique = sorted(np.unique(x_cats))
        y_unique = sorted(np.unique(y_cats))
        
        # Thresholds baseados nas proporções marginais cumulativas
        x_thresholds = []
        for i in range(len(x_unique) - 1):
            p = np.sum(x_cats <= x_unique[i]) / len(x_cats)
            if p > 0.001 and p < 0.999:
                x_thresholds.append(norm.ppf(p))
            elif p <= 0.001:
                x_thresholds.append(-3.0)
            else:
                x_thresholds.append(3.0)
        
        y_thresholds = []
        for i in range(len(y_unique) - 1):
            p = np.sum(y_cats <= y_unique[i]) / len(y_cats)
            if p > 0.001 and p < 0.999:
                y_thresholds.append(norm.ppf(p))
            elif p <= 0.001:
                y_thresholds.append(-3.0)
            else:
                y_thresholds.append(3.0)
        
        # Função objetivo: maximizar verossimilhança
        def negative_log_likelihood(rho):
            """Função de verossimilhança negativa para otimização"""
            if abs(rho) >= 0.99:
                return 1e10
            
            try:
                log_lik = 0.0
                n_total = len(x_cats)
                
                for i, x_val in enumerate(x_unique):
                    for j, y_val in enumerate(y_unique):
                        count = np.sum((x_cats == x_val) & (y_cats == y_val))
                        if count > 0:
                            # Limites da região
                            x_low = x_thresholds[i-1] if i > 0 else -np.inf
                            x_high = x_thresholds[i] if i < len(x_thresholds) else np.inf
                            y_low = y_thresholds[j-1] if j > 0 else -np.inf
                            y_high = y_thresholds[j] if j < len(y_thresholds) else np.inf
                            
                            # Probabilidade da região sob normal bivariada
                            # P(x_low < X <= x_high, y_low < Y <= y_high)
                            p_region = (bivariate_normal_cdf(x_high, y_high, rho) -
                                      bivariate_normal_cdf(x_low, y_high, rho) -
                                      bivariate_normal_cdf(x_high, y_low, rho) +
                                      bivariate_normal_cdf(x_low, y_low, rho))
                            
                            p_region = max(1e-10, min(1.0, p_region))
                            
                            if p_region > 0:
                                log_lik += count * np.log(p_region)
                
                return -log_lik
            except:
                return 1e10
        
        # Otimiza
        result = minimize_scalar(negative_log_likelihood, 
                               bounds=(-0.95, 0.95), 
                               method='bounded',
                               options={'maxiter': 150, 'xatol': 1e-4})
        
        if result.success and not np.isnan(result.x):
            return result.x
        else:
            # Fallback para Spearman
            return rho_init
            
    except Exception as e:
        # Fallback: usa correlação de Spearman
        try:
            return stats.spearmanr(x_clean, y_clean)[0]
        except:
            return np.nan


def tetrachoric_correlation(x, y):
    """
    Calcula correlação tetraicórica entre duas variáveis dicotômicas.
    
    Parâmetros:
    -----------
    x, y : array-like
        Variáveis dicotômicas (0 ou 1)
    
    Retorna:
    --------
    float
        Correlação tetraicórica estimada
    """
    # Remove valores missing
    mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = np.array(x[mask])
    y_clean = np.array(y[mask])
    
    if len(x_clean) < 10:
        return np.nan
    
    # Tabela de contingência 2x2
    n11 = np.sum((x_clean == 1) & (y_clean == 1))
    n10 = np.sum((x_clean == 1) & (y_clean == 0))
    n01 = np.sum((x_clean == 0) & (y_clean == 1))
    n00 = np.sum((x_clean == 0) & (y_clean == 0))
    
    n = n11 + n10 + n01 + n00
    
    if n == 0:
        return np.nan
    
    # Proporções
    p11 = n11 / n
    p10 = n10 / n
    p01 = n01 / n
    p00 = n00 / n
    
    # Marginais
    p1x = (n11 + n10) / n
    p0x = (n01 + n00) / n
    p1y = (n11 + n01) / n
    p0y = (n10 + n00) / n
    
    # Evita divisão por zero
    if p1x == 0 or p0x == 0 or p1y == 0 or p0y == 0:
        return np.nan
    
    # Calcula correlação tetraicórica usando máxima verossimilhança
    try:
        from scipy.stats import norm
        
        # Thresholds (pontos de corte) baseados nas proporções marginais
        z1 = norm.ppf(max(0.001, min(0.999, p1x)))
        z2 = norm.ppf(max(0.001, min(0.999, p1y)))
        
        # Função objetivo: maximizar verossimilhança
        def negative_log_likelihood(rho):
            """Verossimilhança negativa para correlação tetraicórica"""
            if abs(rho) >= 0.99:
                return 1e10
            
            try:
                # Probabilidades sob normal bivariada padrão
                # P(X=1, Y=1) = Φ_2(z1, z2; rho)
                # Onde Φ_2 é a CDF da normal bivariada
                
                # Aproximação usando integração numérica
                # Para implementação completa, usaria scipy.stats.multivariate_normal
                # Aqui usamos aproximação de Olsson (1979)
                
                # Aproximação de Pearson melhorada
                if abs(rho) < 0.01:
                    p11_est = p1x * p1y
                else:
                    # Usa aproximação baseada em expansão de série
                    # Fórmula de aproximação de Pearson
                    p11_est = p1x * p1y + rho * np.sqrt(p1x * p0x * p1y * p0y)
                    p11_est = max(0.0001, min(0.9999, p11_est))
                
                # Verossimilhança
                if p11_est <= 0 or p11_est >= 1:
                    return 1e10
                
                log_lik = (n11 * np.log(p11_est) + 
                          n10 * np.log(max(0.0001, p1x - p11_est)) +
                          n01 * np.log(max(0.0001, p1y - p11_est)) +
                          n00 * np.log(max(0.0001, 1 - p1x - p1y + p11_est)))
                
                return -log_lik
            except:
                return 1e10
        
        # Otimiza
        result = minimize_scalar(negative_log_likelihood, 
                               bounds=(-0.99, 0.99), 
                               method='bounded',
                               options={'maxiter': 200})
        
        if result.success:
            return result.x
        else:
            # Fallback: usa aproximação de Pearson
            if p1x > 0 and p0x > 0 and p1y > 0 and p0y > 0:
                # Fórmula de aproximação de Pearson
                rho_approx = (p11 - p1x * p1y) / np.sqrt(p1x * p0x * p1y * p0y)
                return max(-0.99, min(0.99, rho_approx))
            else:
                return np.nan
                
    except Exception as e:
        # Fallback: usa correlação de Pearson
        try:
            return np.corrcoef(x_clean, y_clean)[0, 1]
        except:
            return np.nan


def ordinal_alpha(df, items):
    """
    Calcula o Alfa Ordinal Generalizado (Ordinal Alpha).
    
    O Alfa Ordinal usa correlações policóricas em vez de correlações de Pearson,
    sendo mais apropriado para dados ordinais.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame com os dados
    items : list
        Lista com os nomes das colunas (itens) a serem avaliados
    
    Retorna:
    --------
    float
        Valor do Alfa Ordinal (0 a 1)
    """
    # Remove linhas com valores missing em qualquer item
    df_clean = df[items].dropna()
    
    if len(df_clean) < 10:
        return np.nan
    
    k = len(items)
    
    # Calcula matriz de correlações policóricas
    polychoric_matrix = np.eye(k)
    
    for i in range(k):
        for j in range(i + 1, k):
            rho = polychoric_correlation(df_clean[items[i]], df_clean[items[j]])
            polychoric_matrix[i, j] = rho
            polychoric_matrix[j, i] = rho
    
    # Verifica se há valores NaN na matriz
    if np.isnan(polychoric_matrix).any():
        # Se houver NaN, tenta usar Spearman como fallback
        corr_matrix = df_clean[items].corr(method='spearman').values
        polychoric_matrix = np.where(np.isnan(polychoric_matrix), corr_matrix, polychoric_matrix)
    
    # Calcula variâncias das correlações policóricas
    # Soma das correlações fora da diagonal
    sum_corr = np.sum(polychoric_matrix) - np.trace(polychoric_matrix)
    mean_corr = sum_corr / (k * (k - 1))
    
    # Fórmula do Alfa Ordinal (similar ao Alfa de Cronbach, mas usando correlações policóricas)
    # Nota: Quando mean_corr <= 0, o alfa pode ser negativo ou não definido
    # Isso indica que as dimensões têm correlações negativas ou muito baixas
    if mean_corr <= 0:
        # Retorna o valor calculado mesmo se negativo (é informativo)
        # Valores negativos indicam que as dimensões são independentes ou negativamente correlacionadas
        ordinal_alpha = (k * mean_corr) / (1 + (k - 1) * mean_corr)
        # Se o resultado for muito extremo (fora de [-1, 1]), retorna NaN
        if ordinal_alpha < -1 or ordinal_alpha > 1:
            return np.nan
        return ordinal_alpha
    
    ordinal_alpha = (k * mean_corr) / (1 + (k - 1) * mean_corr)
    
    return ordinal_alpha


def calculate_ordinal_measures(combined_df):
    """
    Calcula correlações policóricas/tetraicóricas e Alfa Ordinal.
    """
    dimensoes = ['saliencia', 'religiosidade_individual', 'frequencia', 'organizacao']
    
    print("\n" + "=" * 60)
    print("ANÁLISE COM MÉTODOS PARA DADOS ORDINAIS")
    print("=" * 60)
    print("\nCalculando:")
    print("  - Correlações Policóricas/Tetraicóricas")
    print("  - Alfa Ordinal Generalizado")
    print("\nEstes métodos são mais apropriados para dados categóricos/ordinais")
    print("pois assumem variáveis latentes contínuas subjacentes.")
    
    results = []
    
    # Por onda
    print("\n" + "-" * 60)
    print("ALFA ORDINAL POR ONDA:")
    print("-" * 60)
    
    for wave in sorted(combined_df['wave'].unique()):
        df_wave = combined_df[combined_df['wave'] == wave]
        alpha_ord = ordinal_alpha(df_wave, dimensoes)
        
        # Interpretação
        if pd.isna(alpha_ord):
            interp = "Não calculável"
        elif alpha_ord < 0.5:
            interp = "Inadequada"
        elif alpha_ord < 0.7:
            interp = "Aceitável"
        elif alpha_ord < 0.9:
            interp = "Boa"
        else:
            interp = "Excelente"
        
        print(f"{wave:25s}: α_ord = {alpha_ord:.4f} ({interp})")
        results.append({
            'wave': wave,
            'ordinal_alpha': alpha_ord,
            'interpretacao': interp,
            'n': len(df_wave[dimensoes].dropna())
        })
    
    # Conjunto completo
    print("\n" + "-" * 60)
    print("ALFA ORDINAL - CONJUNTO COMPLETO:")
    print("-" * 60)
    
    alpha_ord_total = ordinal_alpha(combined_df, dimensoes)
    
    if pd.isna(alpha_ord_total):
        interp_total = "Não calculável"
    elif alpha_ord_total < 0.5:
        interp_total = "Inadequada"
    elif alpha_ord_total < 0.7:
        interp_total = "Aceitável"
    elif alpha_ord_total < 0.9:
        interp_total = "Boa"
    else:
        interp_total = "Excelente"
    
    print(f"Todas as ondas: α_ord = {alpha_ord_total:.4f} ({interp_total})")
    results.append({
        'wave': 'Todas as ondas',
        'ordinal_alpha': alpha_ord_total,
        'interpretacao': interp_total,
        'n': len(combined_df[dimensoes].dropna())
    })
    
    # Matriz de correlações policóricas
    print("\n" + "-" * 60)
    print("MATRIZ DE CORRELAÇÕES POLICÓRICAS/TETRAICÓRICAS:")
    print("-" * 60)
    print("(Conjunto Completo)")
    
    polychoric_matrix = pd.DataFrame(index=dimensoes, columns=dimensoes)
    
    for i, dim1 in enumerate(dimensoes):
        for j, dim2 in enumerate(dimensoes):
            if i == j:
                polychoric_matrix.loc[dim1, dim2] = 1.0
            else:
                rho = polychoric_correlation(combined_df[dim1], combined_df[dim2])
                polychoric_matrix.loc[dim1, dim2] = rho
    
    # Converte para float
    polychoric_matrix = polychoric_matrix.astype(float)
    print(polychoric_matrix.round(4))
    
    # Salva resultados
    ordinal_df = pd.DataFrame(results)
    ordinal_file = BASE_DIR / 'ordinal_alpha_indice_religiosidade.csv'
    ordinal_df.to_csv(ordinal_file, index=False, encoding='utf-8')
    print(f"\nResultados do Alfa Ordinal salvos em: {ordinal_file}")
    
    polychoric_file = BASE_DIR / 'correlacao_polychoric_dimensoes.csv'
    polychoric_matrix.to_csv(polychoric_file, encoding='utf-8')
    print(f"Matriz de correlações policóricas salva em: {polychoric_file}")
    
    return ordinal_df, polychoric_matrix


def calculate_vif(df, items):
    """
    Calcula o Fator de Inflação da Variância (VIF) para cada variável.
    
    O VIF mede quanto a variância de um coeficiente de regressão aumenta
    devido à multicolinearidade. Valores altos (>10 ou >5) indicam multicolinearidade.
    
    Parâmetros:
    -----------
    df : DataFrame
        DataFrame com os dados
    items : list
        Lista com os nomes das colunas (variáveis) a serem avaliadas
    
    Retorna:
    --------
    dict
        Dicionário com VIF para cada variável
    """
    # Remove linhas com valores missing em qualquer item
    df_clean = df[items].dropna()
    
    if len(df_clean) < len(items) + 10:
        return {item: np.nan for item in items}
    
    vif_dict = {}
    
    for item in items:
        # Variável dependente: item atual
        y = df_clean[item].values
        
        # Variáveis independentes: todas as outras
        X_items = [i for i in items if i != item]
        X = df_clean[X_items].values
        
        # Regressão linear
        try:
            reg = LinearRegression()
            reg.fit(X, y)
            r_squared = reg.score(X, y)
            
            # VIF = 1 / (1 - R²)
            if r_squared >= 1.0:
                vif = np.inf
            elif r_squared >= 0.999:
                vif = 1000.0  # Valor alto para indicar multicolinearidade extrema
            else:
                vif = 1.0 / (1.0 - r_squared)
            
            vif_dict[item] = vif
        except:
            vif_dict[item] = np.nan
    
    return vif_dict


def calculate_condition_number(corr_matrix):
    """
    Calcula o número de condicionamento da matriz de correlação.
    
    O número de condicionamento mede a sensibilidade da matriz a pequenas
    mudanças. Valores altos (>30) indicam multicolinearidade.
    
    Parâmetros:
    -----------
    corr_matrix : array-like ou DataFrame
        Matriz de correlação
    
    Retorna:
    --------
    float
        Número de condicionamento
    """
    try:
        if isinstance(corr_matrix, pd.DataFrame):
            matrix = corr_matrix.values
        else:
            matrix = np.array(corr_matrix)
        
        # Remove NaN e inf
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Calcula número de condicionamento
        cond_number = cond(matrix)
        
        return cond_number
    except:
        return np.nan


def test_multicollinearity(combined_df):
    """
    Testa multicolinearidade entre as dimensões do índice de religiosidade.
    
    Calcula:
    1. VIF (Variance Inflation Factor)
    2. Número de Condicionamento
    3. Determinante da matriz de correlação
    4. Análise de correlações altas
    
    Parâmetros:
    -----------
    combined_df : DataFrame
        DataFrame combinado com todas as ondas
    
    Retorna:
    --------
    dict
        Dicionário com resultados dos testes
    """
    dimensoes = ['saliencia', 'religiosidade_individual', 'frequencia', 'organizacao']
    
    print("\n" + "=" * 60)
    print("TESTE DE MULTICOLINEARIDADE")
    print("=" * 60)
    print("\nA multicolinearidade ocorre quando há alta correlação entre variáveis")
    print("independentes, o que pode causar problemas em análises estatísticas.")
    print("\nMétodos utilizados:")
    print("  1. VIF (Variance Inflation Factor)")
    print("  2. Número de Condicionamento")
    print("  3. Determinante da Matriz de Correlação")
    print("  4. Análise de Correlações Altas")
    
    results = []
    
    # Por onda
    print("\n" + "-" * 60)
    print("MULTICOLINEARIDADE POR ONDA:")
    print("-" * 60)
    
    for wave in sorted(combined_df['wave'].unique()):
        df_wave = combined_df[combined_df['wave'] == wave]
        df_clean = df_wave[dimensoes].dropna()
        
        if len(df_clean) < len(dimensoes) + 10:
            print(f"\n{wave:25s}: Dados insuficientes (n={len(df_clean)})")
            continue
        
        print(f"\n{wave}:")
        print(f"  N = {len(df_clean)}")
        
        # 1. VIF
        vif_dict = calculate_vif(df_wave, dimensoes)
        print("\n  1. VIF (Variance Inflation Factor):")
        print("     Interpretação: VIF > 10 indica multicolinearidade problemática")
        print("                   VIF > 5 indica multicolinearidade moderada")
        for dim, vif in vif_dict.items():
            if pd.isna(vif):
                status = "Não calculável"
            elif vif >= 10:
                status = "⚠️  PROBLEMÁTICA"
            elif vif >= 5:
                status = "⚠️  Moderada"
            elif vif >= 2:
                status = "Aceitável"
            else:
                status = "OK"
            print(f"     {dim:25s}: VIF = {vif:6.2f} ({status})")
        
        # 2. Número de Condicionamento
        corr_matrix = df_clean[dimensoes].corr()
        cond_number = calculate_condition_number(corr_matrix)
        print(f"\n  2. Número de Condicionamento: {cond_number:.2f}")
        if pd.isna(cond_number):
            cond_status = "Não calculável"
        elif cond_number > 30:
            cond_status = "⚠️  PROBLEMÁTICA (>30)"
        elif cond_number > 15:
            cond_status = "⚠️  Moderada (15-30)"
        else:
            cond_status = "OK (<15)"
        print(f"     Interpretação: {cond_status}")
        
        # 3. Determinante da matriz de correlação
        try:
            det_value = det(corr_matrix.values)
            print(f"\n  3. Determinante da Matriz de Correlação: {det_value:.6f}")
            if det_value < 0.01:
                det_status = "⚠️  PROBLEMÁTICA (<0.01)"
            elif det_value < 0.1:
                det_status = "⚠️  Moderada (0.01-0.1)"
            else:
                det_status = "OK (>0.1)"
            print(f"     Interpretação: {det_status}")
        except:
            det_value = np.nan
            print(f"\n  3. Determinante da Matriz de Correlação: Não calculável")
        
        # 4. Análise de correlações altas
        print(f"\n  4. Análise de Correlações Altas:")
        print("     Correlações > 0.7 indicam possível multicolinearidade")
        high_corr_pairs = []
        for i, dim1 in enumerate(dimensoes):
            for j, dim2 in enumerate(dimensoes):
                if i < j:
                    corr_val = corr_matrix.loc[dim1, dim2]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((dim1, dim2, corr_val))
        
        if high_corr_pairs:
            for dim1, dim2, corr_val in high_corr_pairs:
                print(f"     ⚠️  {dim1} ↔ {dim2}: r = {corr_val:.3f}")
        else:
            print("     ✓ Nenhuma correlação > 0.7 encontrada")
        
        # Salva resultados
        results.append({
            'wave': wave,
            'n': len(df_clean),
            'vif_saliencia': vif_dict.get('saliencia', np.nan),
            'vif_religiosidade_individual': vif_dict.get('religiosidade_individual', np.nan),
            'vif_frequencia': vif_dict.get('frequencia', np.nan),
            'vif_organizacao': vif_dict.get('organizacao', np.nan),
            'vif_max': max([v for v in vif_dict.values() if not pd.isna(v)], default=np.nan),
            'vif_mean': np.nanmean(list(vif_dict.values())),
            'condition_number': cond_number,
            'determinant': det_value,
            'high_correlations_count': len(high_corr_pairs)
        })
    
    # Conjunto completo
    print("\n" + "-" * 60)
    print("MULTICOLINEARIDADE - CONJUNTO COMPLETO:")
    print("-" * 60)
    
    df_clean = combined_df[dimensoes].dropna()
    print(f"\nN = {len(df_clean)}")
    
    # 1. VIF
    vif_dict = calculate_vif(combined_df, dimensoes)
    print("\n1. VIF (Variance Inflation Factor):")
    for dim, vif in vif_dict.items():
        if pd.isna(vif):
            status = "Não calculável"
        elif vif >= 10:
            status = "⚠️  PROBLEMÁTICA"
        elif vif >= 5:
            status = "⚠️  Moderada"
        elif vif >= 2:
            status = "Aceitável"
        else:
            status = "OK"
        print(f"   {dim:25s}: VIF = {vif:6.2f} ({status})")
    
    # 2. Número de Condicionamento
    corr_matrix = df_clean[dimensoes].corr()
    cond_number = calculate_condition_number(corr_matrix)
    print(f"\n2. Número de Condicionamento: {cond_number:.2f}")
    if pd.isna(cond_number):
        cond_status = "Não calculável"
    elif cond_number > 30:
        cond_status = "⚠️  PROBLEMÁTICA (>30)"
    elif cond_number > 15:
        cond_status = "⚠️  Moderada (15-30)"
    else:
        cond_status = "OK (<15)"
    print(f"   Interpretação: {cond_status}")
    
    # 3. Determinante
    try:
        det_value = det(corr_matrix.values)
        print(f"\n3. Determinante da Matriz de Correlação: {det_value:.6f}")
        if det_value < 0.01:
            det_status = "⚠️  PROBLEMÁTICA (<0.01)"
        elif det_value < 0.1:
            det_status = "⚠️  Moderada (0.01-0.1)"
        else:
            det_status = "OK (>0.1)"
        print(f"   Interpretação: {det_status}")
    except:
        det_value = np.nan
        print(f"\n3. Determinante da Matriz de Correlação: Não calculável")
    
    # 4. Análise de correlações altas
    print(f"\n4. Análise de Correlações Altas:")
    high_corr_pairs = []
    for i, dim1 in enumerate(dimensoes):
        for j, dim2 in enumerate(dimensoes):
            if i < j:
                corr_val = corr_matrix.loc[dim1, dim2]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((dim1, dim2, corr_val))
    
    if high_corr_pairs:
        for dim1, dim2, corr_val in high_corr_pairs:
            print(f"   ⚠️  {dim1} ↔ {dim2}: r = {corr_val:.3f}")
    else:
        print("   ✓ Nenhuma correlação > 0.7 encontrada")
    
    # Salva resultados
    results.append({
        'wave': 'Todas as ondas',
        'n': len(df_clean),
        'vif_saliencia': vif_dict.get('saliencia', np.nan),
        'vif_religiosidade_individual': vif_dict.get('religiosidade_individual', np.nan),
        'vif_frequencia': vif_dict.get('frequencia', np.nan),
        'vif_organizacao': vif_dict.get('organizacao', np.nan),
        'vif_max': max([v for v in vif_dict.values() if not pd.isna(v)], default=np.nan),
        'vif_mean': np.nanmean(list(vif_dict.values())),
        'condition_number': cond_number,
        'determinant': det_value,
        'high_correlations_count': len(high_corr_pairs)
    })
    
    # Salva resultados
    multicollinearity_df = pd.DataFrame(results)
    multicollinearity_file = BASE_DIR / 'multicolinearidade_indice_religiosidade.csv'
    multicollinearity_df.to_csv(multicollinearity_file, index=False, encoding='utf-8')
    print(f"\nResultados de multicolinearidade salvos em: {multicollinearity_file}")
    
    # Nota sobre interpretação
    print("\n" + "-" * 60)
    print("NOTA SOBRE OS RESULTADOS:")
    print("-" * 60)
    print("Multicolinearidade moderada entre dimensões de religiosidade é esperada")
    print("e não necessariamente problemática, pois as dimensões medem aspectos")
    print("relacionados do mesmo construto (religiosidade).")
    print("\nNo entanto, multicolinearidade extrema (VIF > 10, Cond > 30) pode indicar:")
    print("1. Redundância entre dimensões")
    print("2. Necessidade de revisão do índice")
    print("3. Possível combinação de dimensões altamente correlacionadas")
    
    return multicollinearity_df


def main():
    """Função principal que processa todas as ondas e cria o índice."""
    print("=" * 60)
    print("CONSTRUÇÃO DO ÍNDICE DE RELIGIOSIDADE - WORLD VALUES SURVEY")
    print("=" * 60)
    
    all_results = []
    
    # Processa cada onda
    for wave_name, wave_config in VARIABLES_MAPPING.items():
        result = process_wave(wave_name, wave_config)
        if result is not None:
            all_results.append(result)
    
    if not all_results:
        print("\nErro: Nenhuma onda foi processada com sucesso.")
        return
    
    # Combina todos os resultados
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Estatísticas descritivas
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DESCRITIVAS DO ÍNDICE")
    print("=" * 60)
    
    print("\nPor Onda:")
    stats_by_wave = combined_df.groupby('wave')['indice_religiosidade'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    print(stats_by_wave)
    
    print("\nEstatísticas Gerais:")
    print(f"Total de registros: {len(combined_df)}")
    print(f"Registros com índice completo: {combined_df['indice_religiosidade'].notna().sum()}")
    print(f"Média geral: {combined_df['indice_religiosidade'].mean():.3f}")
    print(f"Desvio padrão: {combined_df['indice_religiosidade'].std():.3f}")
    print(f"Mínimo: {combined_df['indice_religiosidade'].min():.3f}")
    print(f"Máximo: {combined_df['indice_religiosidade'].max():.3f}")
    
    # Salva resultados
    output_file = BASE_DIR / 'indice_religiosidade_completo.csv'
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nResultados salvos em: {output_file}")
    
    # Salva estatísticas por onda
    stats_file = BASE_DIR / 'estatisticas_indice_religiosidade.csv'
    stats_by_wave.to_csv(stats_file, encoding='utf-8')
    print(f"Estatísticas salvas em: {stats_file}")
    
    # Salva resumo por dimensão
    print("\n" + "=" * 60)
    print("RESUMO POR DIMENSÃO")
    print("=" * 60)
    
    dimensoes = ['saliencia', 'religiosidade_individual', 'frequencia', 'organizacao']
    for dim in dimensoes:
        print(f"\n{dim.upper()}:")
        dim_stats = combined_df.groupby('wave')[dim].agg(['count', 'mean', 'std']).round(3)
        print(dim_stats)
    
    # Calcula Alfa de Cronbach
    cronbach_df = calculate_cronbach_alpha(combined_df)
    
    # Calcula medidas ordinais (policóricas e alfa ordinal)
    ordinal_df, polychoric_matrix = calculate_ordinal_measures(combined_df)
    
    # Testa multicolinearidade
    multicollinearity_df = test_multicollinearity(combined_df)
    
    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)


if __name__ == '__main__':
    main()

