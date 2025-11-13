# 1. Clonar o repositório (se for usar git)
git clone <url>
cd <nome>

# 2. Executar Script de instalação 
chmod +x install.sh
./install.sh

# 2. Instalar dependências Python
pip install -r requirements.txt

# 3. Instalar Ollama (se ainda não tiver)
# https://ollama.com/download

# 4. Baixar o modelo que você vai usar
ollama pull llama3

# 5. Rodar o script
python main.py
