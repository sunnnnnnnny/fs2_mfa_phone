# coding:utf-8

import os
import zw_router_test
import logging


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    logging.basicConfig(level=logging.DEBUG)
    handler = logging.FileHandler('flask.log', encoding='UTF-8')
    #handler.setLevel(logging.INFO) # 设置日志记录最低级别为DEBUG，低于DEBUG级别的日志记录会被忽略，不设置setLevel()则默认为NOTSET级别。
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    zw_router_test.tts_app.logger.addHandler(handler)
    zw_router_test.tts_app.run('0.0.0.0', '8512', False)

