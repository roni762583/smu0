# Gemini updated, gpt modified

FROM python:3.10-bullseye

# Copy all project files to /app
COPY . /app

# Set working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
        unzip \
        libglu1-mesa-dev \
        libgl1-mesa-dev \
        libosmesa6-dev \
        xvfb \
        patchelf \
        ffmpeg \
        cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /root/.mujoco \
    && cd /root/.mujoco \
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

# Set LD_LIBRARY_PATH for Mujoco
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin

# Install Python dependencies including JupyterLab
#RUN pip install --upgrade pip \
#    && pip install numpy torch torchvision gym matplotlib jupyterlab

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install numpy torch torchvision matplotlib jupyterlab \
    && pip install gymnasium \
    && pip install ray[default] \
    && pip install v20 \
    && pip install ipywidgets




# Expose the port for JupyterLab
EXPOSE 8888

# Set the entry point to start JupyterLab
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--notebook-dir=/app", "--NotebookApp.default_url=/lab", "--NotebookApp.terminals_enabled=True", "--NotebookApp.token=''", "--NotebookApp.password=''"]
