# 安装环境
# pip install -r requirements.txt 

SCRIPT_DIR="/root/mpdocvqa/script"
# 解压数据
bash $SCRIPT_DIR/unzip_data.sh

WORK_DIR="/root/mpdocvqa"
cd ${WORK_DIR}/src
accelerate launch trainer3.py