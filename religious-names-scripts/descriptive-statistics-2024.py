import pandas as pd

# Load the dataset
df = pd.read_csv('consulta_cand_2024_BRASIL.csv', sep=';', encoding='latin1')
print(df.describe())

# 1. Count the unique numbers of DS_OCUPACAO and the % of the total each one represents
occupation_counts = df['DS_OCUPACAO'].value_counts(normalize=True) * 100
occupation_counts_df = occupation_counts.reset_index()
occupation_counts_df.columns = ['DS_OCUPACAO', 'Percentage']

# Save to CSV
occupation_counts_df.to_csv('occupation_counts.csv', index=False)

# 2. Filter data for DS_OCUPACAO = 'SACERDOTE OU MEMBRO DE ORDEM OU SEITA RELIGIOSA'
religious_filter_df = df[df['DS_OCUPACAO'] == 'SACERDOTE OU MEMBRO DE ORDEM OU SEITA RELIGIOSA']
republicanos_filter_df = df[df['NR_CANDIDATO'] == 10123]

# Save to CSV
religious_filter_df.to_csv('religious_filter.csv', index=False)
republicanos_filter_df.to_csv('republicanos_filter.csv', index=False)

# 3. Search for religious affiliations in NM_URNA_CANDIDATO
religious_terms = ['Pastor', 'Pastora', 'Missionário', 'Missionária', 'Irmão', 'Irmã', 'Reverendo', 'Reverenda', 'Bispo', 'Bispa', 'Apostolo','Apostola', 'de Madureira']

# Filter by religious terms
religious_name_filter = df[df['NM_URNA_CANDIDATO'].str.contains('|'.join(religious_terms), case=False, na=False)]

# Save to CSV
religious_name_filter.to_csv('religious_name_filter.csv', index=False, encoding='utf-8')