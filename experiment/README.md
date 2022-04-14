# experiment
### How to preprocess
```bash
python preprocess.py -i {`ORIGIN_DIR`} -o {`PREPROCESSED_DIR`}
```
### How to start
1. config wandb yaml
eg:
```yaml
epochs:
  value: 800
batch_size:
  value: 32
loss_fn:
  value: 'MSE'
lr:
  value: 0.000007
feature_dim:
  value: 256
visual_freq:
  value: 500
data_dir:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR'
processed_dir:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/SEN12MS_CR_PROCESSED_V2'
media_dir:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/media'
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