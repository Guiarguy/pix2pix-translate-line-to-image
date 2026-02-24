# 使用官方 Python 映像檔
FROM python:3.10-slim

# 建立工作資料夾
WORKDIR /app

# 複製當前所有檔案到容器內部
COPY . /app

# 安裝必要套件
RUN pip install --no-cache-dir -r requirements.txt

# 開放 7860 port（Gradio）
EXPOSE 7860

# 執行主程式
CMD ["python", "script/web_server.py"]
