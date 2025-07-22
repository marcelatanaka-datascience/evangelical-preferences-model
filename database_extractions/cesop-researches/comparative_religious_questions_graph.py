import matplotlib.pyplot as plt
import pandas as pd

# Dados fornecidos
total_ai = {'Contém religião': 877, 'Não contém religião': 3551}
amostra_manual = {'Contém religião': 61, 'Não contém religião': 293}

# Criando DataFrame
df = pd.DataFrame([total_ai, amostra_manual], index=['Total IA', 'Amostra Manual'])

# Convertendo para porcentagem
df_percent = df.div(df.sum(axis=1), axis=0) * 100

# Cores personalizadas
colors = ['#fb6a3d', '#361d20']

# Plotando gráfico
ax = df_percent.plot(kind='bar', figsize=(10, 6), rot=0, color=colors)
plt.title('Comparação de Classificação: IA vs Amostra Manual')
plt.ylabel('Porcentagem (%)')
plt.xlabel('Fonte: Elaboração Própria')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Classificação')

# Adicionando rótulos nas barras
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', label_type='edge', fontsize=10, padding=3)

plt.tight_layout()
plt.show()
