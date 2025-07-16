import pandas as pd
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# CONFIGURAÇÕES
CSV_CAMINHO = "pesquisas_cesop.csv"
PASTA_DESTINO = "questionarios"
BLOCO_PAGINAS = 20
BASE_SITE = "https://www.cesop.unicamp.br"

# ========== FUNÇÕES AUXILIARES ==========

def iniciar_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Ative se quiser rodar oculto
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.page_load_strategy = 'eager'
    return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

def baixar_pdf(url_pdf, destino):
    try:
        response = requests.get(url_pdf, timeout=60)
        with open(destino, 'wb') as f:
            f.write(response.content)
        print(f"📥 PDF salvo em: {destino}")
    except Exception as e:
        print(f"❌ Falha no download de {url_pdf}: {e}")

# ========== EXECUÇÃO ==========

# Cria pasta se não existir
os.makedirs(PASTA_DESTINO, exist_ok=True)

# Carrega CSV com pesquisas
df = pd.read_csv(CSV_CAMINHO)
pesquisas = df.to_dict(orient='records')

# Loop por blocos
for bloco_inicio in range(0, len(pesquisas), BLOCO_PAGINAS):
    bloco_fim = min(bloco_inicio + BLOCO_PAGINAS, len(pesquisas))
    print(f"\n🚀 Processando bloco {bloco_inicio+1} a {bloco_fim}")
    driver = iniciar_driver()

    for pesquisa in pesquisas[bloco_inicio:bloco_fim]:
        numero = str(pesquisa['Número da pesquisa']).zfill(5)
        url = pesquisa['Link da pesquisa']
        destino_arquivo = os.path.join(PASTA_DESTINO, f"quest_{numero}.pdf")

        if os.path.exists(destino_arquivo):
            print(f"✅ PDF já existe: {destino_arquivo}")
            continue

        try:
            driver.set_page_load_timeout(60)
            driver.get(url)
            time.sleep(2.5)

            link_pdf = driver.find_element(By.CSS_SELECTOR, "a.w-button.btn-download")
            href = link_pdf.get_attribute("href")
            if not href.startswith("http"):
                href = BASE_SITE + href

            print(f"🔗 Baixando PDF da pesquisa {numero}")
            baixar_pdf(href, destino_arquivo)

        except (TimeoutException, WebDriverException, NoSuchElementException) as e:
            print(f"⚠️ Erro na pesquisa {numero}: {e}")
            continue

    driver.quit()
    print(f"🛑 Bloco {bloco_inicio+1} a {bloco_fim} finalizado.")
    time.sleep(3)  # Respeita o servidor

print("✅ Todos os questionários processados.")
