from graphviz import Digraph

# Criação do diagrama
dot = Digraph(comment='Modelo de Formação de Preferências Políticas no Eleitor Evangélico')

# Nós
dot.node('A', 'Instituições Religiosas\n(Igrejas, lideranças, mídias,\nregras de conduta)')
dot.node('B', 'Subcultura Evangélica\n(Valores morais,\nidentidade coletiva)')
dot.node('C', 'Subcultura Política Evangélica\n(Códigos de leitura política)')
dot.node('D', 'Heurística de Identidade\n(Atalho cognitivo e afetivo)')
dot.node('E', 'Ação Política\n(Voto, posicionamento, ativismo)')

# Conexões
dot.edge('A', 'B', label='Socialização\nprimária e contínua')
dot.edge('B', 'C', label='Aprendizado\ncultural e reforço institucional')
dot.edge('C', 'D', label='Ativação em\ncontextos políticos')
dot.edge('D', 'E', label='Decisão\nbaseada em heurística')

# Salvar imagem localmente
output_path = 'modelo_formacao_preferencias'
dot.render(output_path, format='png', cleanup=True)

print(f"Diagrama salvo como {output_path}.png")
