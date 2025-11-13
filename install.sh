#!/bin/bash

echo "==== Instalando dependências Python ===="
pip install -r requirements.txt

echo "==== Checando se Ollama está instalado ===="
if ! command -v ollama &> /dev/null
then
    echo "⚠️  Ollama não encontrado. Instale a partir de https://ollama.com/download"
    exit 1
fi

echo "==== Checando se o modelo llama3 está disponível ===="
if ! ollama list | grep -q "llama3"
then
    echo "📥 Baixando modelo 'llama3'..."
    ollama pull llama3
else
    echo "👍 Modelo 'llama3' já está instalado."
fi

echo "==== Testando uma chamada simples ao modelo ===="
echo 'Say "hello"' | ollama run llama3

echo "==== Instalação concluída com sucesso! ===="
