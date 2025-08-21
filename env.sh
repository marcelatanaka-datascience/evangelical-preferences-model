#!/bin/bash

ENV_NAME="doctr-env"

echo "🔧 Criando ambiente virtual em $ENV_NAME..."
python3 -m venv $ENV_NAME

echo "✅ Ativando ambiente virtual e instalando dependências..."
source $ENV_NAME/bin/activate

pip install --upgrade pip setuptools wheel

# Instala PyTorch otimizado para Apple Silicon
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/metal

# Instala DocTR e dependências OCR
pip install doctr pymupdf pillow

echo "📦 Instalando extensão do VSCode para reconhecimento do ambiente..."
code --install-extension ms-python.python > /dev/null 2>&1

echo "🛠️ Criando configuração VSCode com o interpretador correto..."
mkdir -p .vscode
echo "{
  \"python.defaultInterpreterPath\": \"\${workspaceFolder}/$ENV_NAME/bin/python\",
  \"python.analysis.extraPaths\": [
    \"\${workspaceFolder}/$ENV_NAME/lib/python3.9/site-packages\"
  ],
  \"python.terminal.activateEnvironment\": true
}" > .vscode/settings.json

echo "🎉 Ambiente virtual '$ENV_NAME' criado e configurado com sucesso!"