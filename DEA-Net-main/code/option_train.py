import torch,os,sys,torchvision,argparse
import torch,warnings
import json

# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str,default='Automatic detection')

parser.add_argument('--epochs', type=int,default=100)
parser.add_argument('--iters_per_epoch', type=int,default=5000)
parser.add_argument('--finer_eval_step', type=int,default=400000)
parser.add_argument('--eval_freq', type=int,default=1,help='evaluate every N epochs (1=every epoch, 0=only at end)')
parser.add_argument('--max_eval_images', type=int,default=500,help='maximum number of test images for evaluation (0=all)')
parser.add_argument('--bs', type=int,default=16,help='batch size')
parser.add_argument('--num_workers', type=int,default=8,help='number of workers for data loading')
parser.add_argument('--start_lr', default=0.0004, type=float, help='start learning rate')
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
parser.add_argument('--no_lr_sche', action='store_true',help='no lr cos schedule')
parser.add_argument('--use_warm_up', type=bool, default=False, help='using warm up in learning rate')

parser.add_argument('--w_loss_L1', default=1., type=float, help='weight of loss L1')
parser.add_argument('--w_loss_CR', default=0.1, type=float, help='weight of loss CR')

# Model architecture
parser.add_argument('--base_dim', type=int, default=32, help='base dimension of model (32=original, 24=lightweight)')

# Knowledge Distillation
parser.add_argument('--use_kd', action='store_true', help='use knowledge distillation')
parser.add_argument('--teacher_model', type=str, default=None, help='path to teacher model for KD')
parser.add_argument('--w_loss_KD', default=0.5, type=float, help='weight of KD loss')
parser.add_argument('--kd_temperature', default=4.0, type=float, help='temperature for KD')

# Night Scene Optimization
parser.add_argument('--use_night_loss', action='store_true', help='use night-aware loss for training')
parser.add_argument('--use_night_augmentation', action='store_true', help='use night data augmentation')

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--model_name', type=str, default='MDCTDN')
parser.add_argument('--saved_model_dir', type=str, default='saved_model')
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
parser.add_argument('--saved_plot_dir', type=str, default='saved_plot')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

parser.add_argument('--dataset', type=str, default='ITS', choices=['ITS', 'OTS', 'HAZE4K', 'Night'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='../dataset', help='root directory of datasets')

# only need for resume
parser.add_argument('--resume', type=bool,default=False)
parser.add_argument('--pre_trained_model', type=str,default='null')

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'

dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
model_dir = os.path.join(dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

# 自动创建或使用已存在的目录（不再退出）
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

opt.saved_model_dir = os.path.join(model_dir, 'saved_model')
opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
opt.saved_plot_dir = os.path.join(model_dir, 'saved_plot')
opt.saved_infer_dir = os.path.join(model_dir, 'saved_infer')

# 创建子目录（如果不存在）
os.makedirs(opt.saved_model_dir, exist_ok=True)
os.makedirs(opt.saved_data_dir, exist_ok=True)
os.makedirs(opt.saved_plot_dir, exist_ok=True)
os.makedirs(opt.saved_infer_dir, exist_ok=True)

if os.path.exists(model_dir):
    print(f'⚠️  {model_dir} already exists, will overwrite results')
else:
    print(f'✓ Created {model_dir}')

print(opt)
print('model_dir:', model_dir)

with open(os.path.join(model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)