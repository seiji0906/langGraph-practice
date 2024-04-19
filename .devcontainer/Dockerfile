# Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/python:3.10

# 作業ディレクトリの設定
WORKDIR /

# Python 仮想環境のセットアップ
RUN python -m venv .venv
ENV PATH="/workspace/.venv/bin:$PATH"

# 仮想環境を有効にした状態で必要なパッケージをインストール
RUN . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

