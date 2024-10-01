import argparse
from solver import Solver
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def parse():
    parser = argparse.ArgumentParser(description="longformer")
    parser.add_argument('-task_name',default='ner',help='the dataset name of task')
    parser.add_argument('-model_dir',default='train_model1',help='output model weight dir')
    parser.add_argument('-seq_length', type=int, default=100, help='sequence length')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-num_step', type=int, default=100000, help='sequence length')
    parser.add_argument('-lr', type=float,default=0.01, help='learniong rate')
    parser.add_argument('-data_dir',default='data_dir',help='data dir')
    parser.add_argument('-load',action='store_true',help='load pretrained model')
    parser.add_argument('-train', action='store_true',help='whether train the model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-test_sen', action='store_true', help='whether test a sentence')
    parser.add_argument('-text', default='', help='input a sentence')
    parser.add_argument('-valid_path',default='data/',help='validation data path')
    parser.add_argument('-train_path',default='data/',help='training data path')
    parser.add_argument('-test_path',default='data/',help='testing data path')
    #parser.add_argument('-adj_file', default='data/ADJ.json',help='event label co-occurrence matrix path')
    #parser.add_argument('-embed_file', default='data/_label_vec_300.json', help='event label embedding path')
    parser.add_argument('-early_stop', default=100, help='early stop')
    parser.add_argument('-seed', type=int, default=12, help='random seed')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver.test()
    elif args.test_sen:
        solver.test_sent()
