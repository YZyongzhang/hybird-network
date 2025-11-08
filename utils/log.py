import logging
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

log_name = input("please input log name : ")
# 创建文件 handler
file_handler = logging.FileHandler(f"log_{log_name}.txt", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
# 把 handler 加到 logger
logger.addHandler(file_handler)