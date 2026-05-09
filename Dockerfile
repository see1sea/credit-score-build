# 项目完成后预装依赖
# 使用官方 Miniconda 镜像作为基础
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 可选：更换 Conda 源为国内镜像（加速）
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes

# 预安装常用基础包（可选，避免每次都要装）
# 注意：不要在这里创建 env，因为你会挂载 ./data/conda-envs 覆盖它
RUN conda install -y numpy pandas scikit-learn jupyter && \
    conda clean -afy

# 安装 pip 包（可选）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 默认不启动任何服务（由 docker-compose 的 command 覆盖）
CMD ["tail", "-f", "/dev/null"]