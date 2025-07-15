import pandas as pd

# Carregar CSV sujo
df = pd.read_csv("pesquisas_cesop.csv")

# Função para limpar prefixos tipo "Título:\n", "Data:\n", etc.
def limpar_prefixo(celula):
    if isinstance(celula, str) and ":\n" in celula:
        return celula.split(":\n", 1)[1].strip()
    return celula

# Aplicar a limpeza em todas as colunas
df_limpo = df.applymap(limpar_prefixo)

# Salvar CSV limpo
df_limpo.to_csv("pesquisas_cesop_limpas.csv", index=False, encoding='utf-8-sig')
print("✅ CSV limpo salvo como 'pesquisas_cesop_limpas.csv'")
