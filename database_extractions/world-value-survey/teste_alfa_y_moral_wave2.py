"""
Script para calcular o Alfa de Cronbach e Alfa Policórico (Ordinal Alpha)
para o índice Y_moral (variáveis V307, V308, V309, V80) da wave 2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.special import ndtr

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
WAVE2_DIR = BASE_DIR / 'wave-2-1991-br'
INPUT_FILE = 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv'

# Importa funções do script de índice de religiosidade
import sys
sys.path.insert(0, str(BASE_DIR))
try:
    from indice_religiosidade import (
        cronbach_alpha,
        ordinal_alpha,
        polychoric_correlation
    )
except ImportError:
    # Se não conseguir importar, define as funções aqui
    print("⚠️  Não foi possível importar funções. Usando implementações locais.")
    
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
    
    def tetrachoric_correlation(x, y):
        """Calcula correlação tetraicórica para variáveis dicotômicas."""
        # Tabela de contingência 2x2
        n00 = np.sum((x == 0) & (y == 0))
        n01 = np.sum((x == 0) & (y == 1))
        n10 = np.sum((x == 1) & (y == 0))
        n11 = np.sum((x == 1) & (y == 1))
        
        n = n00 + n01 + n10 + n11
        if n == 0:
            return np.nan
        
        # Proporções
        p00 = n00 / n
        p01 = n01 / n
        p10 = n10 / n
        p11 = n11 / n
        
        # Marginais
        p1x = p10 + p11
        p0x = p00 + p01
        p1y = p01 + p11
        p0y = p00 + p10
        
        if p1x == 0 or p0x == 0 or p1y == 0 or p0y == 0:
            return np.nan
        
        # Thresholds
        t1x = norm.ppf(p1x)
        t1y = norm.ppf(p1y)
        
        # Função objetivo
        def objective(rho):
            if abs(rho) >= 0.99:
                return 1e10
            try:
                # Probabilidade conjunta
                p11_est = bivariate_normal_cdf(t1x, t1y, rho)
                p11_est = max(1e-10, min(1.0 - 1e-10, p11_est))
                return -n11 * np.log(p11_est) - n10 * np.log(p1x - p11_est) - \
                       n01 * np.log(p1y - p11_est) - n00 * np.log(1 - p1x - p1y + p11_est)
            except:
                return 1e10
        
        try:
            result = minimize_scalar(objective, bounds=(-0.99, 0.99), method='bounded')
            return result.x if result.success else np.nan
        except:
            return np.nan
    
    def bivariate_normal_cdf(x, y, rho):
        """CDF da normal bivariada padrão."""
        if abs(rho) < 0.01:
            return norm.cdf(x) * norm.cdf(y)
        
        try:
            x_norm = norm.cdf(x)
            y_norm = norm.cdf(y)
            
            if abs(rho) > 0.95:
                return min(x_norm, y_norm) if rho > 0 else max(0, x_norm + y_norm - 1)
            
            phi_x = norm.pdf(x) if not (np.isinf(x) or np.isnan(x)) else 0.0
            phi_y = norm.pdf(y) if not (np.isinf(y) or np.isnan(y)) else 0.0
            
            correction = rho * phi_x * phi_y
            
            if abs(rho) > 0.3 and np.isfinite(x) and np.isfinite(y):
                correction += (rho**2) * phi_x * phi_y * x * y / 2
            
            result = x_norm * y_norm + correction
            return max(0.0, min(1.0, result))
        except:
            return 0.0
    
    def polychoric_correlation(x, y):
        """Calcula correlação policórica entre duas variáveis ordinais."""
        mask = ~(pd.isna(x) | pd.isna(y))
        x_clean = np.array(x[mask])
        y_clean = np.array(y[mask])
        
        if len(x_clean) < 10:
            return np.nan
        
        x_cats = pd.Categorical(x_clean).codes
        y_cats = pd.Categorical(y_clean).codes
        
        if len(np.unique(x_cats)) == 2 and len(np.unique(y_cats)) == 2:
            return tetrachoric_correlation(x_cats, y_cats)
        
        # Para mais categorias, usa Spearman como aproximação
        try:
            rho = stats.spearmanr(x_clean, y_clean)[0]
            return rho if not np.isnan(rho) else np.nan
        except:
            return np.nan
    
    def ordinal_alpha(df, items):
        """Calcula o Alfa Ordinal (policórico)."""
        df_clean = df[items].dropna()
        
        if len(df_clean) < 10:
            return np.nan
        
        k = len(items)
        polychoric_matrix = np.eye(k)
        
        for i in range(k):
            for j in range(i + 1, k):
                rho = polychoric_correlation(df_clean[items[i]], df_clean[items[j]])
                polychoric_matrix[i, j] = rho
                polychoric_matrix[j, i] = rho
        
        if np.isnan(polychoric_matrix).any():
            corr_matrix = df_clean[items].corr(method='spearman').values
            polychoric_matrix = np.where(np.isnan(polychoric_matrix), corr_matrix, polychoric_matrix)
        
        sum_corr = np.sum(polychoric_matrix) - np.trace(polychoric_matrix)
        mean_corr = sum_corr / (k * (k - 1))
        
        if mean_corr <= 0:
            return np.nan
        
        ordinal_alpha = (k * mean_corr) / (1 + (k - 1) * mean_corr)
        
        if ordinal_alpha < -1 or ordinal_alpha > 1:
            return np.nan
        
        return ordinal_alpha

def load_data():
    """Carrega os dados do arquivo com z-scores."""
    file_path = WAVE2_DIR / INPUT_FILE
    
    if not file_path.exists():
        print(f"Arquivo não encontrado: {file_path}")
        return None
    
    print(f"Carregando dados de {file_path}...")
    
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        print(f"Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def check_variables(df, variables):
    """Verifica se as variáveis existem no DataFrame."""
    missing_vars = []
    
    for var in variables:
        if var not in df.columns:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Variáveis não encontradas: {', '.join(missing_vars)}")
        return False
    
    return True

def interpret_alpha(alpha):
    """Interpreta o valor do alfa."""
    if pd.isna(alpha):
        return "Não calculável"
    elif alpha < 0.5:
        return "Inadequada"
    elif alpha < 0.7:
        return "Aceitável"
    elif alpha < 0.7:
        return "Aceitável"
    elif alpha < 0.9:
        return "Boa"
    else:
        return "Excelente"

def calculate_reliability(df, variables):
    """Calcula medidas de confiabilidade."""
    print(f"\n{'='*60}")
    print(f"TESTE DE CONSISTÊNCIA INTERNA DO ÍNDICE Y_moral")
    print(f"{'='*60}\n")
    
    print(f"Variáveis do índice: {', '.join(variables)}")
    
    # Remove linhas com valores missing
    df_clean = df[variables].dropna()
    n_valid = len(df_clean)
    
    print(f"\nCasos válidos: {n_valid} de {len(df)}")
    
    if n_valid < 10:
        print("⚠️  Poucos casos válidos para calcular alfa.")
        return None
    
    # Estatísticas descritivas
    print(f"\n{'─'*60}")
    print("ESTATÍSTICAS DESCRITIVAS DAS VARIÁVEIS:")
    print(f"{'─'*60}\n")
    for var in variables:
        print(f"{var}:")
        print(f"  Média: {df_clean[var].mean():.4f}")
        print(f"  DP: {df_clean[var].std():.4f}")
        print(f"  Mín: {df_clean[var].min():.4f}")
        print(f"  Máx: {df_clean[var].max():.4f}")
        print(f"  Valores únicos: {df_clean[var].nunique()}")
    
    # Matriz de correlação
    print(f"\n{'─'*60}")
    print("MATRIZ DE CORRELAÇÃO DE PEARSON:")
    print(f"{'─'*60}\n")
    corr_matrix = df_clean[variables].corr(method='pearson')
    print(corr_matrix.round(4))
    
    # Alfa de Cronbach
    print(f"\n{'─'*60}")
    print("ALFA DE CRONBACH:")
    print(f"{'─'*60}\n")
    
    alpha_cronbach = cronbach_alpha(df_clean, variables)
    
    if pd.isna(alpha_cronbach):
        print("❌ Não foi possível calcular o Alfa de Cronbach.")
    else:
        interp = interpret_alpha(alpha_cronbach)
        print(f"Alfa de Cronbach (α): {alpha_cronbach:.4f}")
        print(f"Interpretação: {interp}")
        print(f"\nNota: O Alfa de Cronbach assume que as variáveis são contínuas.")
        print(f"Para variáveis ordinais, o Alfa Policórico é mais apropriado.")
    
    # Alfa Policórico (Ordinal Alpha)
    print(f"\n{'─'*60}")
    print("ALFA POLICÓRICO (ORDINAL ALPHA):")
    print(f"{'─'*60}\n")
    
    alpha_ordinal = ordinal_alpha(df_clean, variables)
    
    if pd.isna(alpha_ordinal):
        print("❌ Não foi possível calcular o Alfa Policórico.")
    else:
        interp_ord = interpret_alpha(alpha_ordinal)
        print(f"Alfa Policórico (α_ord): {alpha_ordinal:.4f}")
        print(f"Interpretação: {interp_ord}")
        print(f"\nNota: O Alfa Policórico usa correlações policóricas,")
        print(f"sendo mais apropriado para variáveis ordinais.")
    
    # Salva resultados
    results = {
        'indice': 'Y_moral',
        'variaveis': ', '.join(variables),
        'cronbach_alpha': alpha_cronbach,
        'cronbach_interpretacao': interpret_alpha(alpha_cronbach),
        'ordinal_alpha': alpha_ordinal,
        'ordinal_interpretacao': interpret_alpha(alpha_ordinal),
        'n_casos_validos': n_valid
    }
    
    result_df = pd.DataFrame([results])
    output_file = BASE_DIR / 'alfa_y_moral_wave2.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Resultados salvos em: {output_file}")
    
    return results

def main():
    """Função principal."""
    print("="*60)
    print("TESTE DE CONSISTÊNCIA INTERNA: Y_moral - Wave 2 (1991)")
    print("="*60)
    
    # Carrega dados
    df = load_data()
    if df is None:
        return
    
    # Variáveis originais (não padronizadas) para o teste
    variables = ['V307', 'V308', 'V309', 'V80']
    
    # Verifica se as variáveis existem
    if not check_variables(df, variables):
        return
    
    # Calcula medidas de confiabilidade
    results = calculate_reliability(df, variables)
    
    if results:
        print(f"\n{'='*60}")
        print("Análise concluída com sucesso!")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

