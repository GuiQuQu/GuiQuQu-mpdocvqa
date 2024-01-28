# 一键解压mpdocvqa数据
# DATA_DIR="/home/klwang/code/python/mpdocvqa/data/MPDocVQA"
DATA_DIR="/home/klwang/code/python/mpdocvqa/test-data"
QAS_ZIP_FILE="qas.zip"
IMAGE_TAR_FILE="images.tar.gz"
OCR_TAR_FILE="ocr.tar.gz"

cd $DATA_DIR
if ! command -v unzip &> /dev/null; then
    echo "unzip 命令未找到，正在安装..."
    # 根据系统使用相应的包管理器安装 unzip
    if command -v apt-get &> /dev/null; then
        sudo apt-get install unzip
    elif command -v yum &> /dev/null; then
        sudo yum install unzip
    else
        echo "无法确定包管理器，请手动安装 unzip，并重新运行脚本。"
        exit 1
    fi
fi

# 解压 qas.zip 到当前目录
echo "正在解压 ${QAS_ZIP_FILE}..."
unzip $DATA_DIR/${QAS_ZIP_FILE} 2>/dev/null && rm $DATA_DIR/qas.zip
if [ $? -ne 0 ]; then
    echo -e "解压失败，请检查 ${DATA_DIR}/${QAS_ZIP_FILE} 是否存在。\n"
else
    echo -e "解压完成，删除压缩包${QAS_ZIP_FILE}完成\n"
fi

# 解压 images.tar.gz 到当前目录
echo "正在解压 ${IMAGE_TAR_FILE}..."
# mkdir -p $DATA_DIR/images
tar -zxf $DATA_DIR/$IMAGE_TAR_FILE -C $DATA_DIR 1>/dev/null && rm $DATA_DIR/$IMAGE_TAR_FILE
if [ $? -ne 0 ]; then
    echo -e "解压失败，请检查 ${DATA_DIR}/${IMAGE_TAR_FILE} 是否存在。\n"
else
    echo -e "解压完成，删除压缩包${IMAGE_TAR_FILE}完成\n"
fi

# 解压 ocr.tar.gz 到当前目录
echo "正在解压 ${OCR_TAR_FILE}..."
# mkdir -p $DATA_DIR/ocr
tar -zxf $DATA_DIR/${OCR_TAR_FILE} -C $DATA_DIR 1>/dev/null && rm $DATA_DIR/${OCR_TAR_FILE}
if [ $? -ne 0 ]; then
    echo "解压失败，请检查 ${DATA_DIR}/${OCR_TAR_FILE} 是否存在。"
else
    echo "解压完成，删除压缩包${OCR_TAR_FILE}完成"
fi
