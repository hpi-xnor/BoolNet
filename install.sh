# Installing essentials
apt-get update
apt-get -y install \
    build-essential \
    git \
    python-dev \
    python-setuptools \
    python-pip \
    python-smbus \
    libglib2.0-0 \
    libgl1-mesa-glx \
    vim

# Installing apex
git clone https://github.com/NVIDIA/apex && cd apex && python3 setup.py install

# Installing NVIDIA DALI Library
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# Installing tensorboard X, torchtoolbox and pandas
pip install torchtoolbox tensorboardX pandas

# Installing torch, torchvision
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
