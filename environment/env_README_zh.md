需要安装的包：
Python 3.10 + PyTorch 1.13.1（CUDA 11.7） + VideoMamba（causal-conv1d/mamba） + PyG（torch_geometric 及配套依赖）

torch 2.X会更好，由于历史原因使用PyTorch 1.13.1
---
### 首先确保网络畅通


### 一、环境准备与基础配置
#### 1. 创建并激活conda环境
```bash
# 创建Python 3.10环境，命名为[env_name]
conda create -n [env_name] python=3.10 -y
# 激活环境
conda activate [env_name]
```

#### 2. 配置CUDA环境变量（假设已提前安装cuda-11.8）
```bash
# 永久写入环境变量（避免每次激活都配置）
echo "export CUDA_HOME=/usr/local/cuda-11.8" >> ~/.bashrc
echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
echo "export CPATH=\$CUDA_HOME/include:\$CPATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
# 立即生效环境变量
source ~/.bashrc

# 验证CUDA路径（输出/usr/local/cuda-11.8即为正确）
echo $CUDA_HOME
# 验证nvcc版本（输出V11.8.x，向下兼容11.7）
nvcc -V
```

---

### 二、安装PyTorch 1.13.1（CUDA 11.7）
```bash
# 安装指定版本PyTorch，强制绑定CUDA 11.7（conda会自动处理兼容）
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

---

### 三、安装VideoMamba（causal-conv1d + mamba）
#### 1. 克隆代码仓并修改构建配置
```bash
# 克隆VideoMamba仓库
if [ ! -d "VideoMamba" ]; then
    git clone https://github.com/OpenGVLab/VideoMamba.git
fi
```

#### 2. 修改mamba的构建依赖
```bash
# create and write
vim [/path_to_VideoMamba]/causal-conv1d/pyproject.toml
vim [/path_to_VideoMamba]/mamba/pyproject.toml

# the result:
([env_name])username@hostname:[/path_to_VideoMamba]$ cat VideoMamba/causal-conv1d/pyproject.toml
[build-system]
requires = ["setuptools>=61", "wheel", "torch"]
build-backend = "setuptools.build_meta"
([env_name])username@hostname:[/path_to_VideoMamba]$ cat VideoMamba/mamba/pyproject.toml
[build-system]
requires = ["setuptools>=61", "wheel", "torch"]
build-backend = "setuptools.build_meta"
([env_name])username@hostname:[/path_to_VideoMamba]$ 
```


#### 2. 以开发模式安装（-e）
```bash
# 安装causal-conv1d（--no-build-isolation避免隔离构建环境，确保torch可用）
## If you can not install, you can refer to debug1.md file.
pip install -e ./VideoMamba/causal-conv1d --no-build-isolation
pip install -e ./VideoMamba/mamba --no-build-isolation
```

#### 替代方案
如果无法安装VideoMamba或存在兼容性问题。可以安装mamba或snn替代。

---

### 四、安装PyG及配套依赖
#### 1. 安装基础依赖
```bash
pip install terminaltables lmdb pandas transformers==4.28.0 numpy==1.26.4
```

#### 2. 安装PyG核心库
```bash
export CPATH=/usr/local/cuda-11.8/include:$CPATH # 安装的有/usr/local/cuda-11.8
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install torch_geometric
pip install --verbose torch_scatter 
pip install --verbose torch_sparse
### 如果上面出现问题：pip 从镜像下载了 torch_scatter 的源代码包（tar.gz），在构建 wheel 的 metadata 阶段需要 import torch 来检查版本，但 pip 的 build isolation 机制让这个临时隔离环境里没有 torch，导致 ModuleNotFoundError: No module named 'torch'。
# 通过明确指定 wheel 版本号（2.1.1+pt113cu117）并结合 -f 指向官方 wheel 索引，强制 pip 直接下载已经预编译好的二进制 wheel 文件（.whl），完全跳过从源代码构建和 setup.py 中的 import torch 步骤，从而绕过了隔离环境找不到 torch 的问题，直接完成安装。
pip install torch-scatter==2.1.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install yacs h5py terminaltables tqdm librosa datasets matplotlib fvcore
pip install seaborn

# 安装PyG可能会让PyTorch变为CPU版。验证PyTorch是否为GPU版本（输出True即为正确）
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA版本:', torch.version.cuda); print('GPU可用:', torch.cuda.is_available())"
# 预期输出：
# PyTorch版本: 1.13.1
# CUDA版本: 11.7
# GPU可用: True
```

#### 3. 安装其他依赖
```bash
pip install yacs h5py tqdm librosa datasets matplotlib fvcore
```

---

### 五、最终环境验证
```bash
# 验证核心依赖是否正常
python -c "
import torch
import torch_geometric
import causal_conv1d
import mamba
print('=== 环境验证 ===')
print(f'Torch版本: {torch.__version__} | CUDA可用: {torch.cuda.is_available()} | CUDA版本: {torch.version.cuda}')
print(f'PyG版本: {torch_geometric.__version__}')
print(f'causal-conv1d: 导入成功')
print(f'mamba: 导入成功')
"
```
#### 预期输出
```
=== 环境验证 ===
Torch版本: 1.13.1 | CUDA可用: True | CUDA版本: 11.7
PyG版本: x.x.x（具体版本以安装为准）
causal-conv1d: 导入成功
mamba: 导入成功
```

---

### 常见问题解决
