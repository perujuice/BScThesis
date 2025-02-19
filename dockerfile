# Start from the pre-built OpenPose image (supports CPU and GPU)
FROM cwaffles/openpose

# Set the working directory
WORKDIR /workspace

# Fix missing NVIDIA GPG key issue
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install a newer Python version (3.8)
RUN apt update && apt install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    && apt clean

# Set Python 3.8 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python3 -m pip install --upgrade pip

# Copy the requirements file and install Python dependencies using the updated Python version
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

# Expose JupyterLab port (Optional)
EXPOSE 8888

# Set default command to Bash
CMD ["/bin/bash"]