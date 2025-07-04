import requests
import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

class SimpleDriveScraper:
    def __init__(self, base_url, download_dir="downloads"):
        self.base_url = base_url
        self.download_dir = download_dir
        self.setup_driver()
        
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
    
    def setup_driver(self):
        """Configurar Chrome driver"""
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 15)
    
    def get_year_periods(self):
        """Obter os períodos de anos (ex: 1893-1900, 1901-1910, etc.)"""
        print(f"Acessando: {self.base_url}")
        self.driver.get(self.base_url)
        time.sleep(3)
        
        periods = []
        
        # Procurar por elementos que contenham períodos (ex: "1893 a 1900")
        elements = self.driver.find_elements(By.TAG_NAME, "li")
        elements.extend(self.driver.find_elements(By.TAG_NAME, "div"))
        elements.extend(self.driver.find_elements(By.TAG_NAME, "span"))
        
        for element in elements:
            try:
                text = element.text.strip()
                # Procurar padrões como "1893 a 1900" ou "1901-1910"
                if re.search(r'\b(\d{4})\s*a\s*(\d{4})\b', text):
                    periods.append(element)
                    print(f"Período encontrado: {text}")
            except:
                continue
        
        return periods
    
    def get_year_links(self):
        all_year_links = []
        periods = self.get_year_periods()
        for period in periods:
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", period)
                time.sleep(1)
                period.click()
                time.sleep(3)
                print(f"Explorando período: {period.text}")
            except Exception as e:
                print(f"Erro ao clicar no período: {e}")
                continue

            # Coletar todos os links dos anos ANTES de clicar em qualquer um
            year_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-id], .year-button, .year-link")
            if not year_elements:
                all_elements = self.driver.find_elements(By.TAG_NAME, "a")
                all_elements.extend(self.driver.find_elements(By.TAG_NAME, "div"))
                all_elements.extend(self.driver.find_elements(By.TAG_NAME, "button"))
                for elem in all_elements:
                    try:
                        text = elem.text.strip()
                        if re.match(r'^\d{4}', text):
                            year_elements.append(elem)
                    except Exception:
                        continue

            # Agora, para cada ano, coletar o link (sem clicar ainda)
            try:
                for year_elem in year_elements:
                    try:
                        year_text = year_elem.text.strip()
                        # Tentar obter o link diretamente, se possível
                        url = year_elem.get_attribute("href")
                        if not url:
                            # Se não houver href, clique para obter a URL
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", year_elem)
                            time.sleep(1)
                            year_elem.click()
                            time.sleep(3)
                            url = self.driver.current_url
                            self.driver.back()
                            time.sleep(2)
                        if url and "drive.google.com" in url:
                            all_year_links.append({'year': year_text, 'url': url})
                            print(f"✓ Ano {year_text} encontrado: {url}")
                    except Exception:
                        continue
            except Exception:
                continue
        return all_year_links

    def download_drive_folder(self, drive_url, year):
        """Baixar todos os arquivos de uma pasta do Google Drive"""
        print(f"Acessando pasta do ano {year}: {drive_url}")
        self.driver.get(drive_url)
        time.sleep(5)
        
        # Criar diretório para o ano
        year_dir = os.path.join(self.download_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        files_downloaded = 0
        
        try:
            # Aguardar carregar a lista de arquivos
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-id]")))
            
            # Encontrar todos os arquivos
            file_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-id]")
            print(f"Encontrados {len(file_elements)} arquivos")
            
            for i, element in enumerate(file_elements):
                try:
                    # Obter ID do arquivo
                    file_id = element.get_attribute("data-id")
                    
                    # Pular se não tiver ID válido
                    if not file_id or file_id == "_gd" or len(file_id) < 10:
                        continue
                    
                    # Obter nome do arquivo através de múltiplos métodos
                    filename = self.get_filename_from_element(element, i)
                    
                    # Fazer download usando método alternativo
                    if self.download_file_alternative(file_id, filename, year_dir):
                        files_downloaded += 1
                        print(f"✓ Baixado: {filename}")
                    else:
                        print(f"✗ Erro ao baixar: {filename}")
                    
                    time.sleep(2)  # Pausa entre downloads
                    
                except Exception as e:
                    print(f"Erro ao processar arquivo {i+1}: {e}")
                    continue
            
        except TimeoutException:
            print("Timeout ao aguardar carregamento dos arquivos")
        
        print(f"Total baixado para {year}: {files_downloaded} arquivos")
        return files_downloaded
    
    def get_filename_from_element(self, element, index):
        """Extrair nome do arquivo do elemento"""
        filename = f"arquivo_{index+1}.pdf"
        
        try:
            # Tentar diferentes seletores para encontrar o nome
            selectors = [
                "[aria-label]",
                "[title]", 
                ".a-s-fa-Ha-p",
                "[data-tooltip]",
                "span",
                "div"
            ]
            
            for selector in selectors:
                try:
                    name_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for name_elem in name_elements:
                        name = (name_elem.get_attribute("aria-label") or 
                               name_elem.get_attribute("title") or 
                               name_elem.get_attribute("data-tooltip") or 
                               name_elem.text)
                        
                        if name and len(name.strip()) > 5 and "." in name:
                            filename = name.strip()
                            break
                    
                    if filename != f"arquivo_{index+1}.pdf":
                        break
                except:
                    continue
            
        except Exception as e:
            print(f"Erro ao extrair nome do arquivo: {e}")
        
        return filename
    
    def download_file_alternative(self, file_id, filename, directory):
        """Baixar arquivo usando método alternativo com múltiplas tentativas"""
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if not safe_filename.endswith('.pdf'):
            safe_filename += '.pdf'
        
        filepath = os.path.join(directory, safe_filename)
        
        # Verificar se já existe
        if os.path.exists(filepath):
            print(f"Arquivo já existe: {safe_filename}")
            return True
        
        # Tentar diferentes URLs de download
        download_urls = [
            f"https://drive.google.com/uc?id={file_id}&export=download",
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
            f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
        ]
        
        for url in download_urls:
            try:
                print(f"Tentando baixar com URL: {url}")
                
                # Usar selenium para baixar via navegador
                if self.download_with_selenium(url, filepath):
                    return True
                
                # Tentar com requests
                response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
                
                # Verificar se é uma página HTML (erro)
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    print(f"Recebido HTML em vez de arquivo - tentando próxima URL")
                    continue
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Verificar se o arquivo foi baixado corretamente
                    if os.path.getsize(filepath) > 1000:  # Arquivo deve ter pelo menos 1KB
                        return True
                    else:
                        os.remove(filepath)
                        print(f"Arquivo muito pequeno, removendo...")
                
            except Exception as e:
                print(f"Erro com URL {url}: {e}")
                continue
        
        return False
    
    def download_with_selenium(self, url, filepath):
        """Baixar arquivo usando selenium (para arquivos que requerem interação)"""
        try:
            # Ir para a URL do arquivo
            self.driver.get(url)
            time.sleep(3)
            
            # Procurar por botão de download
            download_selectors = [
                "[aria-label*='Download']",
                "[title*='Download']", 
                ".ndfHFb-c4YZDc-GSQQnc-LgbsSe",
                "[data-tooltip*='Download']"
            ]
            
            for selector in download_selectors:
                try:
                    download_btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                    download_btn.click()
                    time.sleep(5)
                    
                    # Verificar se o arquivo foi baixado
                    if os.path.exists(filepath):
                        return True
                    
                    # Verificar arquivos baixados recentemente no diretório
                    download_files = os.listdir(os.path.dirname(filepath))
                    recent_files = [f for f in download_files if f.endswith('.pdf') and 
                                   os.path.getmtime(os.path.join(os.path.dirname(filepath), f)) > time.time() - 30]
                    
                    if recent_files:
                        # Renomear o arquivo baixado
                        downloaded_file = os.path.join(os.path.dirname(filepath), recent_files[0])
                        os.rename(downloaded_file, filepath)
                        return True
                        
                except:
                    continue
                    
        except Exception as e:
            print(f"Erro no download com selenium: {e}")
        
        return False
    
    def run(self):
        """Executar o scraper"""
        try:
            print("Iniciando scraper...")
            
            # Obter links dos anos
            year_links = self.get_year_links()
            
            if not year_links:
                print("Nenhum link de ano encontrado!")
                return
            
            print(f"Encontrados {len(year_links)} anos para processar")
            
            total_downloaded = 0
            
            # Processar cada ano
            for year_info in year_links:
                year = year_info['year']
                url = year_info['url']
                
                print(f"\n=== Processando ano {year} ===")
                downloaded = self.download_drive_folder(url, year)
                total_downloaded += downloaded
                
                time.sleep(2)
            
            print(f"\n=== Scraping concluído! ===")
            print(f"Total de arquivos baixados: {total_downloaded}")
            
        except Exception as e:
            print(f"Erro durante execução: {e}")
        
        finally:
            self.driver.quit()

# Exemplo de uso
if __name__ == "__main__":
    # Substitua pela URL real da sua página
    base_url = "https://ipib.org.br/o-estandarte/"
    
    scraper = SimpleDriveScraper(base_url, "downloads")
    scraper.run()