#!/usr/bin/env python3
"""
Exploração do arquivo WVS Wave 7 Brazil para identificar variáveis de religião, Q20 e Q21
"""

import pyreadstat
import pandas as pd

def explorar_wvs():
    """Explora o arquivo WVS Wave 7 Brazil."""
    sav_file_path = "WVS_Wave_7_Brazil_Spss_v5.1.sav"
    
    try:
        print(f"Carregando dados de: {sav_file_path}")
        # Tentar com diferentes opções
        try:
            df, meta = pyreadstat.read_sav(sav_file_path, encoding='latin1')
        except:
            try:
                df, meta = pyreadstat.read_sav(sav_file_path, encoding='utf-8')
            except:
                df, meta = pyreadstat.read_sav(sav_file_path, encoding='cp1252')
        print(f"Dados carregados com sucesso! Shape: {df.shape}")
        
        print("\n" + "="*60)
        print("INFORMAÇÕES GERAIS DA PESQUISA")
        print("="*60)
        print(f"Número de casos: {len(df)}")
        print(f"Número de variáveis: {len(df.columns)}")
        
        # Procurar variáveis relacionadas a religião
        print("\n" + "="*60)
        print("PROCURANDO VARIÁVEIS DE RELIGIÃO")
        print("="*60)
        religiao_vars = [col for col in df.columns if 'relig' in col.lower() or 'Q289' in col or 'Q289A' in col]
        print(f"Variáveis encontradas: {religiao_vars}")
        
        # Procurar Q20 e Q21
        print("\n" + "="*60)
        print("PROCURANDO Q20 E Q21")
        print("="*60)
        q20_vars = [col for col in df.columns if 'Q20' in col or 'q20' in col]
        q21_vars = [col for col in df.columns if 'Q21' in col or 'q21' in col]
        print(f"Variáveis Q20 encontradas: {q20_vars}")
        print(f"Variáveis Q21 encontradas: {q21_vars}")
        
        # Se encontrou variável de religião, mostrar detalhes
        if religiao_vars:
            var_religiao = religiao_vars[0]
            print(f"\n" + "="*60)
            print(f"DETALHES DA VARIÁVEL DE RELIGIÃO: {var_religiao}")
            print("="*60)
            print(f"Valores únicos:")
            valores = df[var_religiao].value_counts().sort_index()
            for valor, count in valores.items():
                print(f"  {valor}: {count}")
            
            # Tentar obter labels dos metadados
            if meta and hasattr(meta, 'variable_value_labels'):
                labels = meta.variable_value_labels.get(var_religiao, {})
                if labels:
                    print(f"\nLabels dos valores:")
                    for valor, label in labels.items():
                        count = valores.get(valor, 0)
                        print(f"  {valor}: {label} (n={count})")
        
        # Detalhes de Q20
        if q20_vars:
            var_q20 = q20_vars[0]
            print(f"\n" + "="*60)
            print(f"DETALHES DA VARIÁVEL Q20: {var_q20}")
            print("="*60)
            valores = df[var_q20].value_counts().sort_index()
            for valor, count in valores.items():
                print(f"  {valor}: {count}")
            
            if meta and hasattr(meta, 'variable_value_labels'):
                labels = meta.variable_value_labels.get(var_q20, {})
                if labels:
                    print(f"\nLabels dos valores:")
                    for valor, label in labels.items():
                        count = valores.get(valor, 0)
                        print(f"  {valor}: {label} (n={count})")
        
        # Detalhes de Q21
        if q21_vars:
            var_q21 = q21_vars[0]
            print(f"\n" + "="*60)
            print(f"DETALHES DA VARIÁVEL Q21: {var_q21}")
            print("="*60)
            valores = df[var_q21].value_counts().sort_index()
            for valor, count in valores.items():
                print(f"  {valor}: {count}")
            
            if meta and hasattr(meta, 'variable_value_labels'):
                labels = meta.variable_value_labels.get(var_q21, {})
                if labels:
                    print(f"\nLabels dos valores:")
                    for valor, label in labels.items():
                        count = valores.get(valor, 0)
                        print(f"  {valor}: {label} (n={count})")
        
        # Mostrar todas as variáveis que começam com Q
        print("\n" + "="*60)
        print("TODAS AS VARIÁVEIS QUE COMEÇAM COM Q")
        print("="*60)
        q_vars = [col for col in df.columns if col.startswith('Q')]
        for var in sorted(q_vars):
            print(f"  {var}")
        
        return df, meta
        
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    df, meta = explorar_wvs()

