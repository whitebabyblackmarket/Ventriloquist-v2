Speech to Speech with RAG

How to install and setup:

1. git clone https://github.com/All-About-AI-YouTube/speech-to-rag.git
2. cd dir speech-to-rag
3. pip install -r requirements.txt
4. download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
5. extract checkpoints.zip to speech-to-rag folder
6. on https://huggingface.co/coqui/XTTS-v2 download model
7. place XTTS-v2 folder in speech-to-rag folder
8. In talk3.py (openvoice version) set your reference voice PATH on line 254
9. In xtalk.py (xtts version)
