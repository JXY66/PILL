import argparse
import pandas as pd
import time
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset-dir', default='datasets/yoochoose1_10', help='the dataset directory'
)
parser.add_argument('--embedding-dim', type=int, default=32, help='the embedding size')
parser.add_argument('--num-layers', type=int, default=4, help='the number of layers')
parser.add_argument('--without_intent', type=int, default=0, help='0: including intention; 1: No intention')
parser.add_argument('--without_price', type=int, default=0, help='0: including price; 1: No price')
# TODO:新增自注意力层的参数
parser.add_argument('--nhead', type=int, default=2, help='the number of heads of multi-head attention')
parser.add_argument('--layer', type=int, default=1, help='number of SAN layers')
parser.add_argument('--feedforward', type=int, default=4, help='the multipler of hidden state size')

parser.add_argument(
    '--feat-drop', type=float, default=0.5, help='the dropout ratio for features'
)
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')

parser.add_argument(
    '--batch-size', type=int, default=512, help='the batch size for training'
)
parser.add_argument(
    '--epochs', type=int, default=30, help='the number of training epochs'
)
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-4,
    help='the parameter for L2 regularization',
)
parser.add_argument(
    '--patience',
    type=int,
    default=5,
    help='the number of epochs that the performance does not improves after which the training stops',
)
parser.add_argument(
    '--num-workers',
    type=int,
    default=8,
    help='the number of processes to load the input graphs',
)
parser.add_argument(
    '--valid-split',
    type=float,
    default=None,
    help='the fraction for the validation set',
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    help='print the loss after this number of iterations',
)
args = parser.parse_args()
print(args)


from pathlib import Path
import torch as th
from torch.utils.data import DataLoader
from utils.data.dataset import read_dataset, AugmentedDataset
from utils.data.collate import (
    seq_to_eop_multigraph,
    seq_to_shortcut_graph,
    collate_fn_factory,
)
from utils.train import TrainRunner
from pill import PILL

dataset_dir = Path(args.dataset_dir)
# --dataset-dir datasets/diginetica --embedding-dim 32 --num-layers 4
# --dataset-dir datasets/yoochoose --embedding-dim 32 --num-layers 4
print('reading dataset')
train_sessions, test_sessions, num_items = read_dataset(dataset_dir)

if args.valid_split is not None:
    num_valid = int(len(train_sessions) * args.valid_split)
    test_sessions = train_sessions[-num_valid:]
    train_sessions = train_sessions[:-num_valid]

# 将一条session划分成多条子session(item_id_seq next_item_label)
train_set = AugmentedDataset(train_sessions)
test_set = AugmentedDataset(test_sessions)

# 类别映射字典和价格映射字典
category_dict = {}
price_dict = {}

# 获取新商品和类别ID、价格ID对应字典
if "diginetica" in args.dataset_dir:
    with open("utils/data/niid_2_ncid.txt", 'r') as item_category_f:
        # niid_id : n_category_id
        item_category_lines = item_category_f.readlines()
        for each_line in item_category_lines:
            each_id_line_2_list = each_line.split(',')
            category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()
    with open("utils/data/niid_2_priceid.txt", 'r') as item_price_f:
        item_price_lines = item_price_f.readlines()
        for each_line in item_price_lines:
            each_line_2_list = each_line.split(',')
            price_dict[each_line_2_list[0]] = each_line_2_list[1].strip()
    # 获取类别数量
    df = pd.read_csv("utils/data/niid_2_ncid.txt", delimiter=',', names=['iid', 'cid'])
    # 获取价格数量
    df_price = pd.read_csv("utils/data/niid_2_priceid.txt", delimiter=',', names=['iid', 'pid'])
elif 'yoochoose1_10' in args.dataset_dir or 'yoochoose1_4' in args.dataset_dir:
    with open("datasets/yoochoose1_10/renew_yoo_niid_2_cid.txt", 'r') as item_category_f:
        # niid_id : n_category_id
        item_category_lines = item_category_f.readlines()
        for each_line in item_category_lines:
            each_id_line_2_list = each_line.split(',')
            category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()
    with open("datasets/yoochoose1_10/renew_yoo_niid_2_priceid_dispersed_50.txt", 'r') as item_price_f:
        item_price_lines = item_price_f.readlines()
        for each_line in item_price_lines:
            each_line_2_list = each_line.split(',')
            price_dict[each_line_2_list[0]] = each_line_2_list[1].strip()
    # 获取类别数量
    df = pd.read_csv("datasets/yoochoose1_10/renew_yoo_niid_2_cid.txt", delimiter=',', names=['iid', 'cid'])
    # 获取价格数量
    df_price = pd.read_csv("datasets/yoochoose1_10/renew_yoo_niid_2_priceid_dispersed_50.txt", delimiter=',',
                           names=['iid', 'pid'])
else :
    print("请正确输入数据集文件名称(diginetica/yoochoose1_10/yoochoose1_4)")
    exit(0)

if args.num_layers > 1:
    collate_fn = collate_fn_factory(seq_to_eop_multigraph, seq_to_shortcut_graph,
                                    cate_dict=category_dict, price_dict=price_dict)
else:
    collate_fn = collate_fn_factory(seq_to_eop_multigraph, cate_dict=category_dict, price_dict=price_dict)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
)
# 获取类别数量
num_category = df['cid'].max() + 1
print(num_category)
# 获取价格数量
num_price = df_price['pid'].max() + 1
print(num_price)
start = time.time()

model = PILL(args, num_items, num_category, num_price, args.embedding_dim, args.num_layers, feat_drop=args.feat_drop)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = model.to(device)
print('cuda' if th.cuda.is_available() else 'cpu')
print(model)
runner = TrainRunner(
    model,
    train_loader,
    test_loader,
    device=device,
    lr=args.lr,
    weight_decay=args.weight_decay,
    patience=args.patience,
)

print('start training')
mrr, hit = runner.train(args.epochs, args.log_interval)

print('MRR@20\tHR@20')
print(f'{mrr * 100:.3f}%\t{hit * 100:.3f}%')
end = time.time()
print("run time: %f s" % (end - start))
