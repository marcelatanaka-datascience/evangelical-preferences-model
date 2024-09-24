import pyreadstat
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

# Provide the path to your .sav file
sav_file_path = "00294.sav"

# Ler o arquivo .sav
df, meta = pyreadstat.read_sav(sav_file_path)

# Verificar as primeiras linhas das colunas p19 e religiao para garantir que foram carregadas corretamente
print(df[['p19', 'religiao']].head())

# Definir o mapeamento das categorias de p24 para a nova variável "religion_aggregate"
def map_religion(religiao):
    if religiao in [1.0]:
        return "Doesn't have religion"
    elif religiao in [2.0,3.0]:
        return "Catholics"
    elif religiao in [4.0]:
        return "Others"
    else:
        return "Unknown"  # Caso haja valores fora do esperado

# Aplicar o mapeamento para criar a nova coluna 'religion_aggregate'
df['religion_aggregate'] = df['religiao'].apply(map_religion)


# Cruzar os dados das colunas p19 e religiao, normalizando por coluna (percentual por religião)
cross_tab_percent = pd.crosstab(df['p19'], df['religiao'])

# Exibir os valores em porcentagens
cross_tab_percent = cross_tab_percent.round(2)  # Arredondar para duas casas decimais para facilitar a leitura
print(cross_tab_percent)

# Filtrar os dados para quem optou pela opção 2 em p20
df_filtered = df[df['p19'] == 2]

# Criar a tabela de contingência entre as colunas p20 (apenas opção 2) e religião
contingency_table = pd.crosstab(df_filtered['p19'], df_filtered['religion_aggregate'])

# Aplicar o teste do qui-quadrado para ver se existe uma associação significativa geral
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Exibir os resultados do chi-quadrado
print(f"Qui-quadrado: {chi2}")
print(f"Valor-p: {p_value}")
print(f"Graus de liberdade: {dof}")
print("Frequências esperadas:")
print(expected)

# Interpretação do valor-p do qui-quadrado
if p_value < 0.05:
    print("Há uma diferença estatisticamente significante entre os grupos religiosos para a opção 2 (Chi-Squared Test).")
else:
    print("Não há uma diferença estatisticamente significante entre os grupos religiosos para a opção 2 (Chi-Squared Test).")


# Z-Test for proportions (between two specific groups)
# For example: Catholics (code = 2.0) vs Evangelicals (code = 3.0)
group_totals = df['religion_aggregate'].value_counts()
print(group_totals)

# Obter a contagem de pessoas que escolheram 2 em p20 por grupo
chosen_2_counts = df_filtered['religion_aggregate'].value_counts()

# Garantir que todas as categorias estão presentes, adicionando 0 para as que não têm valores
chosen_2_counts = chosen_2_counts.reindex(group_totals.index, fill_value=0)

# Escolha dois grupos específicos, por exemplo, Católicos e Evangélicos
count = [chosen_2_counts['Catholics'], chosen_2_counts['Others']]  # Aqui você usa os códigos correspondentes para Católicos e Evangélicos
nobs = [group_totals['Catholics'], group_totals['Others']]

# Realizar o teste Z para proporções
stat, p_value_z = proportions_ztest(count, nobs)

# Exibir o resultado do Z-test para proporções
print(f"Z-Estatística: {stat}")
print(f"Valor-p do teste Z: {p_value_z}")

# Interpretação do valor-p do teste Z
if p_value_z < 0.05:
    print("Há uma diferença estatisticamente significante entre as proporções de Católicos e Evangélicos que escolheram a opção 2 (Z-Test).")
else:
    print("Não há uma diferença estatisticamente significante entre as proporções de Católicos e Evangélicos que escolheram a opção 2 (Z-Test).")
