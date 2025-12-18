"""
Script para criar variáveis de ameaça e ordenação temporal na base final consolidada.

Passo 1: Criar variável ano_ord (ordinal)
  - 1991 -> 0
  - 1997 -> 1
  - 2006 -> 2
  - 2014 -> 3
  - 2018 -> 4

Passo 2: Criar variável ameaca_tipo_inst (dicotômica)
  - Se ano_onda = 1991 ou 1997 -> 1
  - Caso contrário -> 0

Passo 3: Criar variável ameaca_tipo_moral (dicotômica)
  - Se ano_onda = 1991 ou 1997 -> 0
  - Caso contrário -> 1
"""

import pandas as pd
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / 'base_dados_final_consolidada.csv'
OUTPUT_FILE = BASE_DIR / 'base_dados_final_consolidada.csv'

def main():
    """Função principal."""
    print("="*70)
    print("CRIAÇÃO DE VARIÁVEIS DE AMEAÇA E ORDENAÇÃO TEMPORAL")
    print("="*70)
    
    # Carregar base consolidada
    print(f"\n[1] Carregando base consolidada...")
    df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8-sig')
    print(f"  ✓ Base carregada: {len(df)} registros, {len(df.columns)} colunas")
    
    # Verificar se ano_onda existe
    if 'ano_onda' not in df.columns:
        print("  ❌ Erro: coluna 'ano_onda' não encontrada!")
        return
    
    # Passo 1: Criar variável ano_ord
    print(f"\n[2] Criando variável ano_ord...")
    
    # Mapeamento de ano para ordem
    ano_ord_map = {
        1991: 0,
        1997: 1,
        2006: 2,
        2014: 3,
        2018: 4
    }
    
    df['ano_ord'] = df['ano_onda'].map(ano_ord_map)
    
    # Verificar se todos os valores foram mapeados
    missing_ano_ord = df['ano_ord'].isna().sum()
    if missing_ano_ord > 0:
        print(f"  ⚠️  Atenção: {missing_ano_ord} registros não foram mapeados")
        print(f"     Anos únicos encontrados: {sorted(df['ano_onda'].unique())}")
    else:
        print(f"  ✓ Variável ano_ord criada com sucesso")
        print(f"     Distribuição:")
        for ano, ordem in sorted(ano_ord_map.items()):
            count = (df['ano_ord'] == ordem).sum()
            print(f"       {ano} (ordem {ordem}): {count} registros")
    
    # Passo 2: Criar variável ameaca_tipo_inst
    print(f"\n[3] Criando variável ameaca_tipo_inst...")
    
    # Se ano_onda = 1991 ou 1997 -> 1, caso contrário -> 0
    df['ameaca_tipo_inst'] = ((df['ano_onda'] == 1991) | (df['ano_onda'] == 1997)).astype(int)
    
    count_1 = (df['ameaca_tipo_inst'] == 1).sum()
    count_0 = (df['ameaca_tipo_inst'] == 0).sum()
    print(f"  ✓ Variável ameaca_tipo_inst criada com sucesso")
    print(f"     ameaca_tipo_inst = 1 (1991 ou 1997): {count_1} registros")
    print(f"     ameaca_tipo_inst = 0 (outros anos): {count_0} registros")
    
    # Passo 3: Criar variável ameaca_tipo_moral
    print(f"\n[4] Criando variável ameaca_tipo_moral...")
    
    # Se ano_onda = 1991 ou 1997 -> 0, caso contrário -> 1
    df['ameaca_tipo_moral'] = ((df['ano_onda'] != 1991) & (df['ano_onda'] != 1997)).astype(int)
    
    count_1 = (df['ameaca_tipo_moral'] == 1).sum()
    count_0 = (df['ameaca_tipo_moral'] == 0).sum()
    print(f"  ✓ Variável ameaca_tipo_moral criada com sucesso")
    print(f"     ameaca_tipo_moral = 1 (2006, 2014 ou 2018): {count_1} registros")
    print(f"     ameaca_tipo_moral = 0 (1991 ou 1997): {count_0} registros")
    
    # Verificar consistência entre as variáveis
    print(f"\n[5] Verificando consistência...")
    
    # Verificar se ameaca_tipo_inst e ameaca_tipo_moral são complementares
    inconsistent = ((df['ameaca_tipo_inst'] == 1) & (df['ameaca_tipo_moral'] == 1)).sum()
    if inconsistent > 0:
        print(f"  ⚠️  Atenção: {inconsistent} registros têm ambas variáveis = 1")
    else:
        print(f"  ✓ Variáveis são mutuamente exclusivas (correto)")
    
    # Verificar se há casos onde ambas são 0
    both_zero = ((df['ameaca_tipo_inst'] == 0) & (df['ameaca_tipo_moral'] == 0)).sum()
    if both_zero > 0:
        print(f"  ⚠️  Atenção: {both_zero} registros têm ambas variáveis = 0")
    else:
        print(f"  ✓ Todas as observações estão classificadas (correto)")
    
    # Salvar base atualizada
    print(f"\n[6] Salvando base atualizada...")
    df.to_csv(OUTPUT_FILE, sep=';', index=False, encoding='utf-8-sig')
    print(f"  ✓ Base salva: {OUTPUT_FILE}")
    print(f"     Total de registros: {len(df)}")
    print(f"     Total de colunas: {len(df.columns)}")
    
    # Resumo final
    print(f"\n{'='*70}")
    print("RESUMO DAS NOVAS VARIÁVEIS:")
    print("="*70)
    print(f"\nano_ord:")
    print(f"  Tipo: Ordinal (0-4)")
    print(f"  Valores únicos: {sorted(df['ano_ord'].unique())}")
    print(f"  Missing: {df['ano_ord'].isna().sum()}")
    
    print(f"\nameaca_tipo_inst:")
    print(f"  Tipo: Dicotômica (0/1)")
    print(f"  Valores: {sorted(df['ameaca_tipo_inst'].unique())}")
    print(f"  Missing: {df['ameaca_tipo_inst'].isna().sum()}")
    print(f"  Distribuição:")
    for val in sorted(df['ameaca_tipo_inst'].unique()):
        count = (df['ameaca_tipo_inst'] == val).sum()
        pct = (count / len(df)) * 100
        anos = df[df['ameaca_tipo_inst'] == val]['ano_onda'].unique()
        print(f"    {val}: {count} registros ({pct:.2f}%) - Anos: {sorted(anos)}")
    
    print(f"\nameaca_tipo_moral:")
    print(f"  Tipo: Dicotômica (0/1)")
    print(f"  Valores: {sorted(df['ameaca_tipo_moral'].unique())}")
    print(f"  Missing: {df['ameaca_tipo_moral'].isna().sum()}")
    print(f"  Distribuição:")
    for val in sorted(df['ameaca_tipo_moral'].unique()):
        count = (df['ameaca_tipo_moral'] == val).sum()
        pct = (count / len(df)) * 100
        anos = df[df['ameaca_tipo_moral'] == val]['ano_onda'].unique()
        print(f"    {val}: {count} registros ({pct:.2f}%) - Anos: {sorted(anos)}")
    
    print(f"\n{'='*70}")
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*70)

if __name__ == "__main__":
    main()



