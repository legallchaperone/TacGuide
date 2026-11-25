#!/bin/bash
# Stage 0 on a fresh AutoDL host, with the fixes learned on host 1 (driver 595.71.05, where the RTX renderer segfaulted).
# Usage: tmux new -d -s s0 'bash /root/autodl-tmp/stage0_all.sh 2>&1 | tee /root/autodl-tmp/logs/stage0_all.log'
# Stops early (exit 20) if the renderer gate fails, before the Isaac Lab install.
set -x
D=/root/autodl-tmp
mkdir -p $D/logs $D/tmp
export DEBIAN_FRONTEND=noninteractive TERM=xterm

# 1 record environment
{ nvidia-smi; head -4 /etc/os-release; df -h / $D; hostname; } > $D/logs/00_env.log 2>&1

# 2 graphics + X11 libs (libSM etc. are needed by omni.usd.libs), tmux, Vulkan check
apt-get update -qq
apt-get install -y -qq libvulkan1 vulkan-tools libegl1 cmake build-essential tmux gdb \
  libsm6 libice6 libxt6 libxrender1 libxext6 libgl1 libglu1-mesa libxrandr2 libxi6 libxcursor1 libxinerama1 libx11-6 libglib2.0-0
vulkaninfo --summary > $D/logs/02_vulkan.log 2>&1
grep -q "RTX 4090" $D/logs/02_vulkan.log || { echo "VULKAN_GATE_FAIL"; exit 10; }

# 3 caches onto the data disk (idempotent)
source /root/miniconda3/etc/profile.d/conda.sh
source /etc/network_turbo; export no_proxy=$no_proxy,mirrors.aliyun.com,aliyun.com
export OMNI_KIT_ACCEPT_EULA=YES TMPDIR=$D/tmp
mkdir -p $D/conda/envs $D/conda/pkgs $D/pip-cache $D/cache $D/ov
conda config --add envs_dirs $D/conda/envs
conda config --add pkgs_dirs $D/conda/pkgs
if [ ! -L ~/.cache ]; then
  [ -d ~/.cache ] && cp -a ~/.cache/. $D/cache/ && rm -rf ~/.cache
  ln -s $D/cache ~/.cache
fi
mkdir -p ~/.local/share
[ -L ~/.local/share/ov ] || ln -s $D/ov ~/.local/share/ov
grep -q OMNI_KIT_ACCEPT_EULA ~/.bashrc || echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc

# 4 env, torch (domestic mirror; the official index stalls through the proxy), Isaac Sim 4.5
conda env list | grep -q OpenViTac || conda create -n OpenViTac python=3.10 -y || exit 1
conda activate OpenViTac || exit 1
pip config set global.cache-dir $D/pip-cache
pip install --upgrade 'pip<26' 'setuptools<82' wheel || exit 2
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 -f https://mirrors.aliyun.com/pytorch-wheels/cu124/ || exit 2
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com || exit 3

# 4b renderer gate: bare SimulationApp must start and render headless
cat > $D/gate_render.py <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
for _ in range(10):
    app.update()
print("GATE_OK", flush=True)
app.close()
PY
cd $D && timeout 600 python gate_render.py > $D/logs/08_gate.log 2>&1
grep -q GATE_OK $D/logs/08_gate.log || { echo "RENDER_GATE_FAIL (see logs/08_gate.log)"; exit 20; }

# 5 Isaac Lab 2.1.1 (shallow clone with retries; flatdict needs old setuptools and no build isolation)
if [ ! -d $D/IsaacLab/.git ]; then
  for i in 1 2 3 4 5; do
    git clone --depth 1 --branch v2.1.1 https://github.com/isaac-sim/IsaacLab $D/IsaacLab && break
    rm -rf $D/IsaacLab; sleep 10
  done
fi
cd $D/IsaacLab || exit 5
pip install flatdict==4.0.1 --no-build-isolation || exit 6
./isaaclab.sh --install || exit 6
# isaaclab.sh replaces torch (2.7.0+cu128, then newer); pin it back last
pip uninstall -y torch torchvision torchaudio
pip list 2>/dev/null | awk '{print $1}' | grep -E '^nvidia-.*(cu11|cu13)$|^nvidia-(cublas|cuda-cupti|cuda-nvrtc|cuda-runtime|cufft|cufile|curand|cusolver|cusparse|nvjitlink|nvtx)$|^cuda-(bindings|pathfinder|toolkit)$' | xargs -r pip uninstall -y
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 -f https://mirrors.aliyun.com/pytorch-wheels/cu124/ || exit 7
pip cache purge; rm -rf $D/tmp/*
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h / $D

# 6 no-camera example
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless --max_iterations 5 > $D/logs/06_ant.log 2>&1
echo "ANT_EXIT=$?" | tee -a $D/logs/06_ant.log

# 7 camera example (loops forever, so stop it after 5 minutes); images land in scripts/tutorials/04_sensors/output/camera
timeout -s INT 300 python scripts/tutorials/04_sensors/run_usd_camera.py --enable_cameras --headless --save > $D/logs/07_camera.log 2>&1
echo "CAM_EXIT=$?" | tee -a $D/logs/07_camera.log
ls scripts/tutorials/04_sensors/output/camera 2>/dev/null | wc -l
echo STAGE0_ALL_DONE
