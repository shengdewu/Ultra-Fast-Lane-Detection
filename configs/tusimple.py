# DATA
dataset='Tusimple'
data_root = 'F:/ultra-source'

# TRAIN
epoch = 100
batch_size = 8
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9
decay_rate = 0.9

scheduler = 'cos'     #['multi', 'cos', 'expone']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = False

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = 'F:/Ultra-Fast-Lane-Detection/tmp/log'
out_path = 'F:/Ultra-Fast-Lane-Detection/tmp/img'
# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None