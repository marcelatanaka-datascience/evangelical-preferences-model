#!/usr/bin/env python3
"""
Script one-shot para converter TODOS os .sav/.SAV em CSV
na pasta atual `spss_files` (e subpastas).

Uso:
  Estando em `database_extractions/spss_files/`:

      python3 run_once_all_sav_to_csv.py

Ele irá:
- Encontrar todos os arquivos .sav/.SAV em `spss_files` (recursivamente)
- Gerar um CSV para cada arquivo, na subpasta `csv/`
"""

from pathlib import Path

from sav_to_csv import iter_sav_files, convert_one


def main() -> None:
    base_dir = Path(__file__).parent.resolve()
    input_path = base_dir
    out_dir = base_dir / "csv"

    sav_files = iter_sav_files(input_path)
    if not sav_files:
        raise SystemExit(f"Nenhum arquivo .sav encontrado em: {input_path}")

    print(f"Encontrados {len(sav_files)} arquivos .sav em {input_path}")
    for i, sav_path in enumerate(sav_files, 1):
        print(f"[{i}/{len(sav_files)}] Convertendo: {sav_path.relative_to(base_dir)}")
        convert_one(sav_path, out_dir)

    print(f"Concluído. CSVs em: {out_dir}")


if __name__ == "__main__":
    main()

