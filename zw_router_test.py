# coding:utf-8

import json
import os
import time
import datetime
import utility_debug
import logging
from flask import make_response, Response, Flask, jsonify, request
from synthesize_pinyin import text_to_speech

tts_app = Flask(__name__)


@tts_app.route('/', methods=['GET'])
def index():
    resp = make_response('这是一个简单的http服务')
    resp.headers['Content-type'] = 'application/json;charset=utf-8'
    resp.status_code = 200

    return resp


@tts_app.route('/api/v1.0/text2speech/<btxt>', methods=['GET'])
def text2speech(btxt):
    '''
    文本合成为语音
    :param t_type: 合成要使用的模型 date, digit ,numerical_value,person_name
    :param txt: 要合成的文本
    :return: 合成的录音数据(如果成功)，或者错误信息(如果失败)
    '''
    t0 = time.time()
    txt = btxt
    pitch_control, energy_control, duration_control = 1.0, 1.0, 1.0
    error_info = {'code': '0', 'msg': 'ok'}
    tts_app.logger.info("待合成文本为: {}".format(txt))

    # to do 合成
    wav_path = None
    try:
        #wav_path = utility.models["universal"].text2speech(txt, pitch_control, energy_control, duration_control)
        #wav_path = text_to_speech(txt, utility_debug.models["model"], utility_debug.models["vocoder"])
        wav_path = text_to_speech(txt, utility_debug.models)
    except Exception as e:
        print("生成出问题")
        tts_app.logger.error("模型推理报错")
        pass
    if wav_path:
        wav_path = wav_path[0]
    if wav_path and os.path.exists(wav_path):
        mp3_path = wav_path.replace(".wav", ".mp3")
        try:
            """
            os.system(
                "ffmpeg -i {0}  -filter:a \"atempo=1.1\"  -af silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB:stop_periods=-1:stop_duration=0:stop_threshold=-50dB -vn {1} -y -loglevel quiet".format(
                    wav_path, mp3_path))
            """
            os.system("ffmpeg -i {0} -vn {1} -y -loglevel quiet".format(wav_path, mp3_path))
        except:
            tts_app.logger.error("mp3文件转mp3文件失败")
            error_info["code"] = '60001'
            error_info["msg"] = "wav文件转mp3文件失败"
            return Response(json.dumps(error_info), status=200, mimetype='application/json;charset=utf-8')
        if os.path.exists(wav_path):
            #pass
            os.remove(wav_path)
        tts_app.logger.info('{}  :  mp3文件路径为{}'.format(txt, mp3_path))
        try:
            data = None
            with open(mp3_path, 'rb') as fs:
                data = fs.read()
            if data:
                tts_app.logger.info("合成总耗时为：{}".format(str(round(time.time()-t0, 2))))
                return Response(data, status=200, mimetype='audio/mp3')
            else:
                error_info['msg'] = '合成文件{}为空文件'.format(wav_path)
                tts_app.logger.error('合成文件{}为空文件'.format(wav_path))
        except:
            error_info['code'] = '50015'
            error_info['msg'] = u'合成文件{}受损'.format(mp3_path)
            tts_app.logger.error('合成文件{}受损'.format(mp3_path))
    else:
        error_info['code'] = '404001'
        error_info['msg'] = u'未找到合成文件 {}'.format(wav_path)
        tts_app.logger.error('未找到合成文件 {}'.format(wav_path))
    tts_app.logger.info("合成总耗时为：{}".format(str(round(time.time() - t0, 2))))
    return Response(json.dumps(error_info), status=200, mimetype='application/json;charset=utf-8')
