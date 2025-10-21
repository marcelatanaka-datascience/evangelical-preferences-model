#!/usr/bin/env python3
"""
Exploração da Pesquisa 00018 - Verificar variáveis disponíveis
=============================================================

Este script explora todas as variáveis da pesquisa 00018 para identificar
qual variável contém informações sobre religião.
"""

import pyreadstat
import pandas as pd

def explorar_pesquisa():
    """Explora todas as variáveis da pesquisa 00018."""
    sav_file_path = "database_extractions/spss_files/00499.sav"
    
    try:
        print(f"Carregando dados de: {sav_file_path}")
        df, meta = pyreadstat.read_sav(sav_file_path)
        print(f"Dados carregados com sucesso! Shape: {df.shape}")
        
        print("\n" + "="*60)
        print("INFORMAÇÕES GERAIS DA PESQUISA")
        print("="*60)
        print(f"Número de casos: {len(df)}")
        print(f"Número de variáveis: {len(df.columns)}")
        
        print("\n" + "="*60)
        print("VARIÁVEIS DISPONÍVEIS")
        print("="*60)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n" + "="*60)
        print("DETALHES DE CADA VARIÁVEL")
        print("="*60)
        
        for col in df.columns:
            print(f"\n--- {col} ---")
            print(f"Tipo: {df[col].dtype}")
            print(f"Valores únicos: {df[col].nunique()}")
            print(f"Valores nulos: {df[col].isnull().sum()}")
            
            # Mostrar valores únicos
            valores_unicos = df[col].value_counts().sort_index()
            print(f"Valores únicos:")
            for valor, count in valores_unicos.head(10).items():
                print(f"  {valor}: {count}")
            if len(valores_unicos) > 10:
                print(f"  ... e mais {len(valores_unicos) - 10} valores")
            
            # Tentar obter labels dos metadados
            if meta and hasattr(meta, 'variable_value_labels'):
                labels = meta.variable_value_labels.get(col, {})
                if labels:
                    print(f"Labels dos valores:")
                    for valor, label in labels.items():
                        count = valores_unicos.get(valor, 0)
                        print(f"  {valor}: {label} (n={count})")
        
        # Verificar se há metadados de variáveis
        if meta:
            print("\n" + "="*60)
            print("METADADOS DISPONÍVEIS")
            print("="*60)
            
            if hasattr(meta, 'variable_labels'):
                print("Labels das variáveis:")
                for var, label in meta.variable_labels.items():
                    print(f"  {var}: {label}")
            
            if hasattr(meta, 'variable_value_labels'):
                print(f"\nLabels de valores disponíveis para {len(meta.variable_value_labels)} variáveis")
        
        return df, meta
        
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None

if __name__ == "__main__":
    df, meta = explorar_pesquisa()
