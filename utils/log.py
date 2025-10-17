import logging
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

# 创建文件 handler
file_handler = logging.FileHandler("log.txt", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
# 把 handler 加到 logger
logger.addHandler(file_handler)