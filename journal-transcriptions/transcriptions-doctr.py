# OCR com DocTR para PDFs escaneados com layout complexo
# Requisitos: doctr[torch], pymupdf, pillow


import fitz  # PyMuPDF
import os
from PIL import Image
import numpy as np
import tempfile
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# === Configurações ===
USE_GOOGLE_DRIVE = True
GOOGLE_DRIVE_FOLDER_ID = '1-4u2pFzpG4xTIJCGZcgneLER6DyTRUUf'
LOCAL_PDF_FOLDER = './pdfs'



# === OCR com DocTR ===
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# === Função para converter páginas de PDF em imagens PIL ===
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# === Função para estimar número de colunas com base na largura ===
def estimate_columns(image):
    width, _ = image.size
    if width > 1800:
        return 3
    elif width > 1200:
        return 2
    else:
        return 1

# === Função para dividir imagem em colunas verticalmente ===
def split_columns(image, n_cols):
    width, height = image.size
    col_width = width // n_cols
    columns = []
    for i in range(n_cols):
        box = (i * col_width, 0, (i + 1) * col_width, height)
        columns.append(image.crop(box))
    return columns

# === Função de OCR com DocTR ===
def run_ocr_doctr(pdf_path):
    images = pdf_to_images(pdf_path)

    temp_paths = []
    col_counts = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for page_idx, img in enumerate(images):
            n_cols = estimate_columns(img)
            col_counts.append(n_cols)
            columns = split_columns(img, n_cols=n_cols)
            for col_idx, col_img in enumerate(columns):
                img_path = os.path.join(tmpdirname, f"page_{page_idx+1}_col_{col_idx+1}.png")
                col_img.convert('RGB').save(img_path, format='PNG')
                temp_paths.append(img_path)

        doc = DocumentFile.from_images(temp_paths)
        result = model(doc)
        out_text = []

        page_counter = 0
        for page_idx, n_cols in enumerate(col_counts):
            for col_idx in range(n_cols):
                page = result.pages[page_counter]
                out_text.append(f"--- Página {page_idx + 1}, Coluna {col_idx + 1} ---")
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        out_text.append(line_text)
                out_text.append("\n")
                page_counter += 1

        return "\n".join(out_text)

# === Execução principal ===
if __name__ == '__main__':
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

    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Processando: {filename}")
        full_text = run_ocr_doctr(pdf_path)
        output_txt = os.path.join(pdf_folder, filename.replace('.pdf', '.txt'))
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Transcrição salva em: {output_txt}\n")

