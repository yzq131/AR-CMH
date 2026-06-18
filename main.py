import os
import argparse

from load_data import load_dataset, split_dataset, allocate_dataset, Dataset_Config
from defense_model_hash import DAE

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', dest='dataset', default='FLICKR', choices=['FLICKR', 'COCO', 'NUS'],
                    help='name of the dataset')
parser.add_argument('--dataset_path', dest='dataset_path', default='/home/workspace/Y/Dataset/',
                    help='path of the dataset')
parser.add_argument('--defense_method', dest='defense_method', default="CPAH")
parser.add_argument('--defense_models_path', dest='defense_models_path', default='/home/workspace/Y/attacked_models/',
                    help='path of attacked models')
parser.add_argument('--attack_method', dest='attack_method', default='PGD')
parser.add_argument('--attack_methods_path', dest='attack_methods_path', default='/home/workspace/Y/attack_methods/',
                    help='path of attacked models')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--gpu', dest='gpu', type=str, default='0')
parser.add_argument('--bit', dest='bit', type=int, default=32)

parser.add_argument('--batch_size', dest='batch_size', type=int, default=24)

parser.add_argument('--sub_task', dest='sub_task', type=int, default=10)
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=10)
parser.add_argument('--n_epochs_decay', type=int, default=0,
                    help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')

parser.add_argument('--learning_rate_i', dest='lri', type=float, default=1e-3, help='FLICKR NUS 0.001 COCO 0.0001')
parser.add_argument('--learning_rate_t', dest='lrt', type=float, default=1e-3, help='FLICKR NUS 0.001 COCO 0.0001')

parser.add_argument('--lr_policy', type=str, default='linear',
                    help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=10,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--output_path', dest='output_path', default='outputs/', help='models are saved here')
parser.add_argument('--output_dir', dest='output_dir', default='output000', help='the name of output')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

Dcfg = Dataset_Config(args.dataset, args.dataset_path)
X, Y, L = load_dataset(Dcfg.data_path)
X, Y, L = split_dataset(X, Y, L, Dcfg.query_size, Dcfg.training_size, Dcfg.database_size)
Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L = allocate_dataset(X, Y, L)

model = DAE(args=args, Dcfg=Dcfg)

model.train(Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)

