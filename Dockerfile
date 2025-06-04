from ghcr.io/eternalai-org/vibe-browser-use-template:latest

copy requirements.txt requirements.txt
run pip install -r requirements.txt

workdir /workspace

copy app app
copy system_prompt.txt system_prompt.txt
copy server.py server.py
copy scripts scripts

expose 6080
expose 80

entrypoint ["python", "-O", "server.py"]