#!/bin/bash
# Stage 1 install on the sim host (no data collection): cuRobo, TacEx core, UIPC, following OpenViTac scripts/install.sh
# but with Isaac Sim / Isaac Lab already installed by stage0_all.sh. Each step logs a STEP marker; exits non-zero on failure.
set -x
D=/root/autodl-tmp
R=$D/OpenViTac
source /root/miniconda3/etc/profile.d/conda.sh
source /etc/network_turbo; export no_proxy=$no_proxy,mirrors.aliyun.com,aliyun.com,mirrors.tuna.tsinghua.edu.cn,tsinghua.edu.cn
export OMNI_KIT_ACCEPT_EULA=YES TMPDIR=$D/tmp TERM=xterm DEBIAN_FRONTEND=noninteractive
mkdir -p $TMPDIR
TORCH_PIN="torch==2.5.1+cu124 torchvision==0.20.1+cu124 -f https://mirrors.aliyun.com/pytorch-wheels/cu124/"

echo "STEP env_update"; date
for i in 1 2 3; do conda env update -n OpenViTac --file $D/uipc_env_tuna.yaml && break; sleep 15; done; conda list -n OpenViTac | grep -q "^cuda-toolkit" || exit 11
conda activate OpenViTac || exit 1
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib
pip install --upgrade "pip<26" "setuptools<82" wheel pkgconfig
command -v ffmpeg >/dev/null 2>&1 || conda install -n OpenViTac --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main ffmpeg -y
apt-get install -y -qq git-lfs curl zip unzip tar pkg-config ninja-build >/dev/null 2>&1
df -h $D | tail -1

echo "STEP curobo"; date
if ! pip show nvidia_curobo >/dev/null 2>&1; then
  mkdir -p $R/third_party && cd $R/third_party
  if [ ! -d curobo/.git ]; then
    for i in 1 2 3 4 5; do git clone https://github.com/NVlabs/curobo.git && break; rm -rf curobo; sleep 10; done
  fi
  cd curobo && git checkout 0a50de1ba72db304195d59d9d0b1ed269696047f || exit 12
  pip install warp-lang==1.0.0 --no-build-isolation || exit 12
  pip install -e . --no-build-isolation || exit 12
fi
python -c "import torch;print('torch after curobo', torch.__version__)"

echo "STEP tacex_core"; date
if ! pip show tacex >/dev/null 2>&1; then
  cd $R/third_party/TacEx && ./tacex.sh -i || exit 13
  pip uninstall -y torch_scatter
  pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html || exit 13
fi
python -c "import torch;print('torch after tacex', torch.__version__)"

echo "STEP vcpkg"; date
if [ ! -x $D/Toolchain/vcpkg/vcpkg ]; then
  mkdir -p $D/Toolchain && cd $D/Toolchain
  [ -d vcpkg/.git ] || for i in 1 2 3 4 5; do git clone https://github.com/microsoft/vcpkg.git && break; rm -rf vcpkg; sleep 10; done
  cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics || exit 14
fi
export CMAKE_TOOLCHAIN_FILE="$D/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
grep -q CMAKE_TOOLCHAIN_FILE ~/.bashrc || echo "export CMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE" >> ~/.bashrc
export VCPKG_DEFAULT_BINARY_CACHE=$D/cache/vcpkg-bin; mkdir -p $VCPKG_DEFAULT_BINARY_CACHE
export VCPKG_DOWNLOADS=$D/cache/vcpkg-dl; mkdir -p $VCPKG_DOWNLOADS

echo "STEP libuipc"; date
if ! pip show libuipc >/dev/null 2>&1 && ! pip show pyuipc >/dev/null 2>&1; then
  cd $R/third_party/TacEx
  rm -rf source/tacex_uipc/build
  pip install -e source/tacex_uipc -v --no-build-isolation || exit 15
fi

echo "STEP extras"; date
pip install transforms3d trimesh tetgen || exit 16
# make sure nothing moved torch
python - <<'PY' || pip install $TORCH_PIN
import torch, sys
sys.exit(0 if torch.__version__ == "2.5.1+cu124" else 1)
PY
python -c "import torch;print('torch final', torch.__version__, torch.cuda.is_available())"
pip cache purge; rm -rf $D/tmp/*
df -h / $D
echo STAGE1_INSTALL_DONE
