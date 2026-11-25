# source this to get the OpenViTac runtime environment
D=/root/autodl-tmp
source /root/miniconda3/etc/profile.d/conda.sh
conda activate OpenViTac
export OMNI_KIT_ACCEPT_EULA=YES TMPDIR=$D/tmp TERM=xterm
export CUDA_HOME=$CONDA_PREFIX
export CMAKE_TOOLCHAIN_FILE=$D/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake
cd $D/OpenViTac
export LD_LIBRARY_PATH=/root/autodl-tmp/ngx_lib:$LD_LIBRARY_PATH  # NVIDIA NGX core (DLSS); container does not ship it
