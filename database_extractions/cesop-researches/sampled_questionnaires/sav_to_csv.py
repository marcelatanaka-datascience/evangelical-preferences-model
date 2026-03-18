#!/usr/bin/env python3
"""
Convert SPSS .sav / .SAV files to CSV.

Usage examples:
  python sav_to_csv.py "path/to/file.sav"
  python sav_to_csv.py "path/to/file.sav" --outfile "meu_arquivo.csv"
  python sav_to_csv.py "path/to/FOLDER_WITH_SAVs"
  python sav_to_csv.py "path/to/FOLDER_WITH_SAVs" --out "path/to/output"

Notes:
- Output CSV uses UTF-8.
- Each input file becomes one CSV with the same base name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyreadstat


def iter_sav_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".sav":
            raise ValueError(f"Arquivo não suportado: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {input_path}")

    # Accept both .sav and .SAV (suffix check is case-insensitive)
    return sorted(
        [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".sav"]
    )


def convert_one(sav_path: Path, out_dir: Path, outfile: str | None = None) -> None:
    df, meta = pyreadstat.read_sav(str(sav_path))

    # Always write data frame.
    if outfile is not None:
        out_csv = out_dir / outfile
    else:
        out_csv = out_dir / f"{sav_path.stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .sav/.SAV files to CSV.")
    parser.add_argument(
        "input",
        type=str,
        help="Path to a .sav/.SAV file or a folder containing SPSS files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory. Defaults to a sibling folder named 'csv'.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Nome do arquivo CSV de saída (apenas quando 'input' for um único .sav).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir: Path
    if args.out:
        out_dir = Path(args.out).expanduser().resolve()
    else:
        # If input is a file: create ./csv next to it
        if input_path.is_file():
            out_dir = input_path.parent / "csv"
        else:
            out_dir = input_path / "csv"

    sav_files = iter_sav_files(input_path)
    if not sav_files:
        raise SystemExit(f"Nenhum arquivo .sav encontrado em: {input_path}")

    print(f"Encontrados {len(sav_files)} arquivos .sav em {input_path}")

    # Se o usuário passou --outfile mas há mais de um .sav, é ambíguo.
    if args.outfile and len(sav_files) > 1:
        raise SystemExit(
            "O parâmetro --outfile só pode ser usado quando 'input' for um único arquivo .sav."
        )

    for i, sav_path in enumerate(sav_files, 1):
        print(f"[{i}/{len(sav_files)}] Convertendo: {sav_path.name}")
        # Usa outfile apenas se for um único arquivo e o usuário tiver definido.
        outfile = args.outfile if len(sav_files) == 1 else None
        convert_one(sav_path, out_dir, outfile=outfile)

    print(f"Concluído. CSVs em: {out_dir}")


if __name__ == "__main__":
    main()

