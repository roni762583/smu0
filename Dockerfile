FROM python:3.10-bullseye

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
        swig \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /root/.mujoco \
    && cd /root/.mujoco \
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

# Set LD_LIBRARY_PATH for Mujoco
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin

# Copy only requirements.txt first for cache optimization
COPY requirements.txt /tmp/

# Install Python dependencies (use cached pip packages)
RUN pip install --upgrade pip \
    && pip install --cache-dir=/tmp/pip_cache -r /tmp/requirements.txt
    

# Copy the rest of the application code to the container
COPY . /app

# Run your Python script in the background and then keep the container alive
CMD ["sh", "-c", "python run_trading_game.py & tail -f /dev/null"]

