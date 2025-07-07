import pandas as pd
import matplotlib.pyplot as plt

# Dados preparados
dfMdP = pd.DataFrame({
    "Year": [1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
             2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
             2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    "Values": [0.10, 0.00, 0.20, 0.20, 0.35, 0.30, 0.90, 0.10, 0.00, 0.10,
               0.30, 0.40, 0.00, 0.50, 0.60, 0.40, 0.60, 1.10, 0.40, 1.00,
               0.80, 0.90, 1.30, 0.80, 0.30, 0.30, 0.50, 0.60],
    "Equality": [0.10, 0.00, 0.00, 0.70, 0.60, 0.20, 0.10, 0.10, 0.10, 0.10,
                 0.20, 0.50, 0.40, 0.20, 0.20, 0.00, 0.10, 0.10, 0.20, 0.10,
                 0.20, 0.00, 0.10, 0.00, 0.00, 0.00, 0.10, 0.00]
})
dfOJB = pd.DataFrame({
    "Year": [1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
             2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
             2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
    "Values": [0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.20, 0.20, 0.00, 0.10,
               0.10, 0.00, 0.00, 0.20, 0.10, 0.00, 0.00, 0.20, 0.00, 0.90,
               0.30, 0.00, 0.10, 0.20, 0.30, 0.00, 0.10, 0.00],
    "Equality": [0.00, 0.00, 0.00, 0.30, 0.40, 0.20, 0.90, 0.00, 0.20, 0.00,
                 0.00, 0.00, 0.20, 0.20, 0.10, 0.20, 0.00, 0.00, 0.20, 0.00,
                 0.10, 0.10, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00]
})

# Define categorias de eixo x com placeholder "No data"
years_pre = [str(y) for y in dfMdP.Year if y <= 1991]
years_post = [str(y) for y in dfMdP.Year if y >= 2001]
categories = years_pre + ["No data"] + years_post

def plot_panel(ax, df_pub, title, color_vals, color_eq):
    # posições x
    x_pre = [categories.index(str(y)) for y in df_pub.Year if y <= 1991]
    x_post = [categories.index(str(y)) for y in df_pub.Year if y >= 2001]
    df1 = df_pub[df_pub.Year <= 1991]
    df2 = df_pub[df_pub.Year >= 2001]
    
    # plot Values e Equality com cores fixas
    ax.plot(x_pre, df1.Values, marker='o', linestyle='-', color='#361d20', label='Valores')
    ax.plot(x_post, df2.Values, marker='o', linestyle='-', color='#361d20')
    ax.plot(x_pre, df1.Equality, marker='s', linestyle='--', color='#fb6a3d', label='Igualdade')
    ax.plot(x_post, df2.Equality, marker='s', linestyle='--', color='#fb6a3d')
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45)
    ax.set_title(title)
    ax.grid(False)
    ax.set_xlabel('Ano')

# plot vertical (2 painéis)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
colors = ('tab:blue', 'tab:orange')  # valores, igualdade

plot_panel(axes[0], dfMdP, 'Mensageiro da Paz', *colors)
plot_panel(axes[1], dfOJB, 'O Jornal Batista', *colors)

axes[0].set_ylabel('Artigos por edição')
axes[1].set_ylabel('Artigos por edição')

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(loc='upper left')

# legenda única
fig.suptitle('Valores tradicionais e igualdade religiosa em publicações evangélicas no Brasil (1982–2018)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # deixa espaço embaixo
fig.text(0.5, 0.01,
         "Fonte: elaboração própria a partir dos dados de Boas (2023, p. 123 e 138)",
         ha='center', va='bottom', fontsize=9)
plt.show()

