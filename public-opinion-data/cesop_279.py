import pyreadstat
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

# Provide the path to your .sav file
sav_file_path = "00279.sav"

# Ler o arquivo .sav
df, meta = pyreadstat.read_sav(sav_file_path)

# Verificar as primeiras linhas das colunas p18 e p24 para garantir que foram carregadas corretamente
print(df[['p18', 'p24']].head())

# Definir o mapeamento das categorias de p24 para a nova variável "religion_aggregate"
def map_religion(p24_value):
    if p24_value in [0.0, 1.0]:
        return "Doesn't have religion"
    elif p24_value in [2.0, 3.0]:
        return "Catholics"
    elif p24_value in [4.0, 5.0]:
        return "Evangelicals"
    elif p24_value in [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
        return "Others"
    else:
        return "Unknown"  # Caso haja valores fora do esperado

# Aplicar o mapeamento para criar a nova coluna 'religion_aggregate'
df['religion_aggregate'] = df['p24'].apply(map_religion)

# Verificar as primeiras linhas para garantir que o mapeamento foi aplicado corretamente
print(df[['p24', 'religion_aggregate']].head())

# Exibir as contagens de cada grupo religioso
print(df['religion_aggregate'].value_counts())

# Cruzar os dados das colunas p18 e religion_aggregate
cross_tab_counts = pd.crosstab(df['p18'], df['religion_aggregate'])
print("Crosstab of counts:")
print(cross_tab_counts)

# Filtrar os dados para quem optou pela opção 2 em p18
df_filtered = df[df['p18'] == 2]

# Criar a tabela de contingência entre as colunas p18 (apenas opção 2) e religião
contingency_table = pd.crosstab(df_filtered['p18'], df_filtered['religion_aggregate'])

# Aplicar o teste do qui-quadrado
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Exibir os resultados
print(f"Qui-quadrado: {chi2}")
print(f"Valor-p: {p_value}")
print(f"Graus de liberdade: {dof}")
print("Frequências esperadas:")
print(expected)

# Calcular os valores para o Z-test de proporções
# Obter os totais de cada grupo e o número de pessoas que escolheram 2 em p18
group_totals = df['religion_aggregate'].value_counts()

# Obter a contagem de pessoas que escolheram 2 em p18 por grupo
chosen_2_counts = df_filtered['religion_aggregate'].value_counts()

# Garantir que todas as categorias estão presentes, adicionando 0 para as que não têm valores
chosen_2_counts = chosen_2_counts.reindex(group_totals.index, fill_value=0)

# Preparar os dados para o teste Z
count = [chosen_2_counts['Catholics'], chosen_2_counts['Evangelicals']]
nobs = [group_totals['Catholics'], group_totals['Evangelicals']]

# Realizar o teste Z para proporções
stat, p_value_z = proportions_ztest(count, nobs)

# Exibir o resultado do Z-test para proporções
print(f"Z-Estatística: {stat}")
print(f"Valor-p do teste Z: {p_value_z}")

# Interpretação do valor-p
if p_value_z < 0.05:
    print("Há uma diferença estatisticamente significante entre as proporções de Católicos e Evangélicos que afirmam que o casamento entre pessoas do mesmo sexo nao deve ser reconhecido.")
else:
    print("Não há uma diferença estatisticamente significante entre as proporções de Católicos e Evangélicos que afirmam que o casamento entre pessoas do mesmo sexo nao deve ser reconhecido.")
