#coding:utf-8
import yaml
import torch
from utils.model  import get_model, get_vocoder
from g2pW.get_convert import conv


dataset = "csmsc"
dataset = "AISHELL3"
dataset = "term_mini"
dataset = "dongd_mini"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocess_config= "config/{}/preprocess.yaml".format(dataset)
model_config = "config/{}/model.yaml".format(dataset)
train_config = "config/{}/train.yaml".format(dataset)
preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

restore_step=200000
model = get_model(restore_step, configs, device, train=False)
vocoder = get_vocoder(model_config, device)


models = {
          'model': model,
          'vocoder':vocoder,
          'g2p':conv
}

