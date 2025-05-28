from python:3.12-slim
env DEBIAN_FRONTEND=noninteractive

run apt-get update && apt-get install -y --no-install-recommends \
    git \
    xvfb \
    x11vnc \
    openbox \
    procps \
    xdg-utils \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

run git clone https://github.com/novnc/noVNC.git /opt/novnc && \
    git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify && \
    chmod +x /opt/novnc/utils/novnc_proxy

copy requirements.txt requirements.base.txt .
env PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
run pip install -r requirements.base.txt && patchright install chromium --no-shell --with-deps
run pip install -r requirements.txt

workdir /workspace

copy app app
copy system_prompt.txt system_prompt.txt
copy server.py server.py
copy scripts scripts

expose 6080
expose 80

env DISPLAY=:99
env XDG_SESSION_TYPE=x11
env GDK_SCALE=1
env GDK_DPI_SCALE=1
env BROWSER_WINDOW_SIZE_WIDTH=1440
env BROWSER_WINDOW_SIZE_HEIGHT=1440
env SCREEN_COLOR_DEPTH_BITS=24
env NO_VNC_PORT=6080
env CHROME_DEBUG_PORT=9222

env OPENAI_BASE_URL="http://localmodel:65534/v1"
env OPENAI_API_KEY="no-need"

env LLM_BASE_URL="http://localmodel:65534/v1"
env LLM_API_KEY="no-need"
env LLM_MODEL_ID="no-need"
env IN_DOCKER=1

entrypoint ["python", "-O", "server.py"]