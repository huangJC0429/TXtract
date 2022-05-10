# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口
    model='TXtract'

    pretrained_bert_name = 'bert-base-chinese'

    # pickle_path = './Data/ALL_PKL文件' # 包含某一属性值的所有样例的title、attibute、value、title-sequence-label编码
    pickle_path = './Data/精简_PKL文件'
    test_pickle_path = './Data/test_data/test.pkl'
    load_model_path = './checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载
    test = False# True代表测试模型，False代表训练模型

    batch_size = 32  # batch size
    embedding_dim = 768
    hidden_dim = 200
    cond_att_dim = 50
    tagset_size = 4
    use_gpu = True  # user GPU or not

    num_workers = 0  # how many workers for loading data
    print_freq = 500  # print info every N batch

    max_epoch = 20
    lr = 2e-5  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # L2正则
    dropout = 0.2
    seed = 1234
    device = 'cuda'

    # for poincare
    poincare_size = 50  # poincare embedding size
    num_product = 576 # 所有级目录种类的个数
    gamma = 0.5  # 权重系数，控制两个任务的loss
    w = 0.5

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        https://www.programiz.com/python-programming/methods/built-in/hasattr
        https://www.programiz.com/python-programming/methods/built-in/setattr
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
