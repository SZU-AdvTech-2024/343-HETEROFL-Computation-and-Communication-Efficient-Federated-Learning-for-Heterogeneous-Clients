import yaml

global cfg
if 'cfg' not in globals():
    with open('config.yml', 'r') as f:
        #读入config.yml，其内容适配为一个字典，存在cfg变量中
        cfg = yaml.load(f, Loader=yaml.FullLoader)