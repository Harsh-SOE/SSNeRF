# 1. Force Conda to use strict channel priority to avoid mixed-channel dependency hell
conda config --set channel_priority strict

# 2. Install the completely matching compiler and CUDA toolchain directly into the sandbox
# This installs PyTorch (CUDA 12.1), CUDA Toolkit (12.1), and GCC/G++ (v12) into your env.
conda install -y -c pytorch -c nvidia -c conda-forge \
    pytorch==2.3.0 \
    pytorch-cuda=12.1 \
    cuda-toolkit=12.1 \
    gcc_linux-64=12 \
    gxx_linux-64=12 \
    ninja \
    cmake

# 3. Handle Python 3.12+ compatibility by falling back to a setuptools version 
# that still includes the 'pkg_resources' module expected by tiny-cuda-nn
pip install "setuptools<70.0.0" wheel

# 4. Strip out any system-level environment overrides and explicitly lock 
# all build tools to the local Conda environment binaries
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

# 5. Fix the linker path so it uses Conda's internal driver stubs (.so files)
# This prevents the final "cannot find -lcuda" error when building without host driver access
export LDFLAGS="-L$CONDA_PREFIX/lib/stubs"

# 6. Auto-detect your specific GPU architecture and pass it to the compiler
# This prevents nvcc from wasting hours compiling for architectures you don't own
export TCNN_CUDA_ARCHITECTURES=$(python -c "import torch; print(torch.cuda.get_device_capability(0)[0]*10 + torch.cuda.get_device_capability(0)[1])")

# 7. Wipe any half-baked compilation artifacts from previous failed attempts
pip cache purge

# 8. Pull, build, and install tiny-cuda-nn directly into your environment
# Bypassing build isolation ensures pip actually uses the toolchain we just injected.
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation