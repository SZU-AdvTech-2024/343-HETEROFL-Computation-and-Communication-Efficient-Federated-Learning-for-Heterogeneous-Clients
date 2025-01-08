import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import random
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed import Federation
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}


def main():
    process_control()
    #处理控制参数
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    #生成随机种子列表，列表长度等于实验数量，每轮实验都有一个自己的种子
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        #生成当前实验轮次的模型标识符
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print("第{i}个模型标识符：{cfg['model_tag']}")
        #用_将模型的标识符连接起来放入cfg中
        print('Experiment: {}'.format(cfg['model_tag']))
        #输出实验信息
        runExperiment()
        #执行实验
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    print("当前种子:{seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #上面生成pytorch和cuda的随机种子
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    #获取数据信息，根据cfg中的data_name和subset
    process_dataset(dataset)
    #处理数据信息
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    #初始化模型和优化器
    #如果存在记录点，即使用resume模式，恢复中断训练
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        #如果不存在恢复，则重新训练
        last_epoch = 1
        #分割数据集
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        #日志记录
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    #如果数据集没有被分割，则进行分割
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    #联邦学习初始化
    global_parameters = model.state_dict()
    federation = Federation(global_parameters, cfg['model_rate'], label_split)
    #每一轮中进行本地训练
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        #进行训练
        train(dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch)

        test_model = stats(dataset['train'], model)
        test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation, epoch)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    start_time = time.time()
    for m in range(num_active_users):
        local_parameters[m] = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        local_time = (time.time() - start_time) / (m + 1)
        epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
        exp_finished_time = epoch_finished_time + datetime.timedelta(
            seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                             'Learning rate: {}'.format(lr),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
        logger.append(info, 'train', mean=False)
        logger.write('train', cfg['metric_name']['train']['Local'])
    federation.combine(local_parameters, param_idx, user_idx)
    global_model.load_state_dict(federation.global_parameters)
    return


def stats(dataset, model):
    with torch.no_grad():
        test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"])'
                          .format(cfg['model_name']))
        test_model.load_state_dict(model.state_dict(), strict=False)
        data_loader = make_data_loader({'train': dataset})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def test(dataset, data_split, label_split, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for m in range(cfg['num_users']):
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
        data_loader = make_data_loader({'test': dataset})['test']
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return


def make_local(dataset, data_split, label_split, federation, epoch):
    model_rates = federation.model_rate  # 假设这是一个 numpy 数组，包含每个用户的 model_rate

    # 获取各个 rate 的用户索引
    rate_1_users = [idx for idx, rate in enumerate(model_rates) if rate == 1.0]
    rate_05_users = [idx for idx, rate in enumerate(model_rates) if rate == 0.5]
    rate_025_users = [idx for idx, rate in enumerate(model_rates) if rate == 0.25]

    # 设置每轮选择的比例
    num_rate_1 = 2  # 从 rate=1 的用户中选择 2 个
    num_rate_05 = 4  # 从 rate=0.5 的用户中选择 4 个
    num_rate_025 = 4  # 从 rate=0.25 的用户中选择 4 个

    # 随机选择用户
    selected_rate_1_users = random.sample(rate_1_users, num_rate_1)
    selected_rate_05_users = random.sample(rate_05_users, num_rate_05)
    selected_rate_025_users = random.sample(rate_025_users, num_rate_025)

    # 合并选择的用户
    user_idx = selected_rate_1_users + selected_rate_05_users + selected_rate_025_users
    print(f"目前是第{epoch}轮，该轮参与训练的节点都有：{user_idx}")

    # 分配本地模型参数
    local_parameters, param_idx = federation.distribute(user_idx)

    # 初始化本地模型
    local = [None for _ in range(num_rate_1 + num_rate_05 + num_rate_025)]
    for m in range(num_rate_1 + num_rate_05 + num_rate_025):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(model_rate_m, data_loader_m, label_split[user_idx[m]])

    return local, local_parameters, user_idx, param_idx
class Local:
    def __init__(self, model_rate, data_loader, label_split):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split

    def train(self, local_parameters, lr, logger):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        local_parameters = model.state_dict()
        return local_parameters


if __name__ == "__main__":
    main()