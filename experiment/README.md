# experiment
### How to preprocess
```bash
python preprocess.py -i {`ORIGIN_DIR`} -o {`PREPROCESSED_DIR`}
```
### How to start
1. config wandb yaml
eg:
```yaml
model:
  value: 'Test'
epochs:
  value: 800
batch_size:
  value: 12 
loss_fn:
  value: 'CharbonnierLoss'
val_percent:
  value: 0.2
# >>> learning rate >>>
scheduler:
  value: GradualWarmupScheduler
lr:
  value: 0.0002
min_lr:
  value: 0.000001
T_max:
  value: 60
# <<< learning rate <<<
feature_dim:
  value: 256
# per batch
visual_freq:
  value: 500
# per epoch
validate_freq:
  value: 1
train_dir:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR_PROCESSED_V4/TRAIN'
test_dir:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR_PROCESSED_V4/TEST'
media_dir:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/media'
checkpoints_dir:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/checkpoints'
in_channels:
  value: 15
feature_dim:
  value: 256
out_channels:
  value: 13
num_layers:
  value: 6
season:
  value: 'ROIs1868_summer'
scene_black_list:
  value: null
scene_white_list:
  value: null
```
file struture like:
```bash
.
├── config-defaults.yaml # here is your config yaml
├── dataset/
├── metrics/
├── models/
├── preprocess/
├── preprocess.py
├── README.md
├── train.py
```

2. setup gpu config
```bash
export CUDA_VISIBLE_DEVICES = '0,1,2'
```

3. run
```bash
python train.py
```