# 基于HeteroFL的复现和物联网功耗限制改进
本项目是基于HeteroFL的改进，在原文的基础，复现了论文结果，同时加入了物联网节点的功耗限制。
想要运行本项目，需要先下载对应的数据集和模型。
原文连接：[HeteroFL]https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients
项目的依赖与原项目相同，并未做修改，但是原项目中 一些依赖已经过期，可以使用新的版本，注意torch版本即可。
数据集在运行程序后会自动下载，也可以手东下载放入dataset文件夹中，所需要的数据集为cifar10，MINST，wikitext2，使用的模型包括resnet18，CNN和transformer，可以等程序下载，也可以自行在hugging face下载放入model文件夹。
自行训练可通过以下指令：
* train表示训练，数据集为MNIST，模型使用cnn，比例为a2，b8，激活率为0.1，共100个节点，fix模式，iid数据
```
python train_classifier_fed.py --data_name MNIST --model_name conv --control_name 1_100_0.1_iid_fix_a2-b8_bn_1_1
```
* test表示测试已经训练好的数据，需已存在checkpoint，数据集为CIFAR10，模型为resnet18，iid数据，激活率0.1，共10个节点，比例为a2,b4,c4，fix模式
```
python test_classifier_fed.py --data_name CIFAR10 --model_name resnet18 --control_name 1_10_0.1_iid-2_fix_a2-b4-c4_gn_0_0
```
训练完的效果，在output文件夹中，可通过tensorboard进行可视化观察。