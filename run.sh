speaker=AISHELL3
python prepare_align.py config/${speaker}/preprocess.yaml
#python preprocess.py config/${speaker}/preprocess.yaml
#python finetune.py -p config/${speaker}/preprocess.yaml -m config/${speaker}/model.yaml -t config/${speaker}/train.yaml
