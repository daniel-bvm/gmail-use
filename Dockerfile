from python:3.11

env DEBIAN_FRONTEND=noninteractive
env DISPLAY=:99
env PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
env XDG_SESSION_TYPE=x11
env GDK_SCALE=1
env GDK_DPI_SCALE=1
env BROWSER_WINDOW_SIZE_WIDTH=1280
env BROWSER_WINDOW_SIZE_HEIGHT=768
env SCREEN_COLOR_DEPTH_BITS=24
env NO_VNC_PORT=6080
env CHROME_DEBUG_PORT=9222

run apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libxss1 \
    libasound2 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libdrm2 \
    libgbm1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libgl1-mesa-glx xvfb x11vnc fluxbox \
    && rm -rf /var/lib/apt/lists/*

run git clone https://github.com/novnc/noVNC.git /opt/novnc && \
    git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify && \
    chmod +x /opt/novnc/utils/novnc_proxy

copy requirements.txt .
run pip install -r requirements.txt --no-cache-dir
run patchright install chromium
workdir /workspace

copy app app
copy system_prompt.txt system_prompt.txt
copy server.py server.py

expose 6080
expose 80

env OPENAI_BASE_URL="http://localmodel:65534/v1"
env OPENAI_API_KEY="no-need"

env LLM_BASE_URL="http://localmodel:65534/v1"
env LLM_API_KEY="no-need"
env LLM_MODEL_ID="no-need"

entrypoint ["python", "-O", "server.py"]