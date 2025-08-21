# OCR com Tesseract para PDFs de imagens escaneadas
# Requisitos: pytesseract, pillow, opencv-python, pymupdf, pydrive2

import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF
import os

# === Configuração ===
USE_GOOGLE_DRIVE = True  # Trocar para False se quiser usar apenas pasta local
GOOGLE_DRIVE_FOLDER_ID = '1-4u2pFzpG4xTIJCGZcgneLER6DyTRUUf'
LOCAL_PDF_FOLDER = './pdfs'
LANGUAGE = 'por'  # usar 'por' se o pacote estiver instalado corretamente

# === Google Drive ===
if USE_GOOGLE_DRIVE:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive

    pdf_folder = './drive_pdfs'
    os.makedirs(pdf_folder, exist_ok=True)

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({
        'q': f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
    }).GetList()

    for file in file_list:
        local_path = os.path.join(pdf_folder, file['title'])
        file.GetContentFile(local_path)
else:
    pdf_folder = LOCAL_PDF_FOLDER

# === Função de pré-processamento da imagem ===
def preprocess_image(pil_img: Image.Image) -> Image.Image:
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    bin_img = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
    coords = np.column_stack(np.where(bin_img > 0))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = bin_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = bin_img
    rgb = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)

# === OCR de uma imagem ===
def ocr_page_image(image: Image.Image) -> str:
    try:
        pre = preprocess_image(image)
        text = pytesseract.image_to_string(pre, lang=LANGUAGE, config='--psm 3')
        return text
    except Exception as e:
        return f"[Erro ao processar imagem: {e}]"

# === OCR de um PDF ===
def ocr_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_pages = []
    for page_number in range(len(doc)):
        try:
            pix = doc[page_number].get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = ocr_page_image(img)
            text_pages.append(f"--- Página {page_number + 1} ---\n{text}\n")
        except Exception as e:
            text_pages.append(f"--- Página {page_number + 1} ---\n[Erro na página: {e}]\n")
    return "\n".join(text_pages)

# === Execução principal ===
if __name__ == '__main__':
    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Processando: {filename}")
        full_text = ocr_pdf(pdf_path)
        output_txt = os.path.join(pdf_folder, filename.replace('.pdf', '.txt'))
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Transcrição salva em: {output_txt}\n")
