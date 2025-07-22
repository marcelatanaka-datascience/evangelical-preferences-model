import os
import re
import json
import time
import signal
import logging
import pandas as pd
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIGURAÇÕES ===
PDF_DIR = "./questionarios"
CSV_PATH = "pesquisas_cesop.csv"
OUTPUT_CSV = "pesquisas_cesop_regex.csv"
LOG_FILE = "analise_pdfs.log"
PROGRESS_FILE = "progresso.json"
WORKERS = 4
BATCH_SIZE = 30

# === CONFIGURAÇÃO DO LOG ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === REGEX PARA "RELIGIÃO" E VARIANTES ===
religiao_regex = re.compile(r"religi[aã]o", re.IGNORECASE)

# === CONTROLE DE PROGRESSO ===
progresso = {"ultimo_idx": -1}

def salvar_progresso():
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progresso, f)

def carregar_progresso():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            progresso.update(json.load(f))

def signal_handler(sig, frame):
    logger.info("Interrompido pelo usuário. Salvando progresso...")
    salvar_progresso()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === FUNÇÃO PRINCIPAL DE BUSCA ===
def verificar_religiao(idx, numero_pesquisa):
    pdf_path = os.path.join(PDF_DIR, f"quest_{str(numero_pesquisa).zfill(5)}.pdf")
    try:
        reader = PdfReader(pdf_path)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text() or ""
        encontrou = bool(religiao_regex.search(texto))
        logger.info(f"[{numero_pesquisa}] {'✔️ Encontrado' if encontrou else '❌ Não encontrado'}")
        return idx, encontrou
    except Exception as e:
        logger.warning(f"[{numero_pesquisa}] Erro ao ler PDF: {e}")
        return idx, False

# === EXECUÇÃO ===
def main():
    carregar_progresso()
    df = pd.read_csv(CSV_PATH)
    if "contém_religião" not in df.columns:
        df["contém_religião"] = ""

    total = len(df)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for i in range(progresso["ultimo_idx"] + 1, total, BATCH_SIZE):
            futures = []
            for j in range(i, min(i + BATCH_SIZE, total)):
                numero = df.at[j, "Número da pesquisa"]
                futures.append(executor.submit(verificar_religiao, j, numero))

            for future in as_completed(futures):
                idx, contem = future.result()
                df.at[idx, "contém_religião"] = contem
                progresso["ultimo_idx"] = idx

            df.to_csv(OUTPUT_CSV, index=False)
            salvar_progresso()

    logger.info("✅ Processamento completo.")
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

if __name__ == "__main__":
    main()
