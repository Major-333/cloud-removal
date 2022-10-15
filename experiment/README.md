# experiment
## Preprocess
Our preprocessing consists of three operations: crop, normalize, and clip.

The file path structure of the dataset before and after preprocessing is the same. They can both be managed by `sen12ms_cr_dataset.dataset.SEN12MSCRDataset`.
The file path structure strictly follows the original file structure of the TUM dataset: https://mediatum.ub.tum.de/1554803

While the file path structure remains the same, the file format is converted from `.tif` to `.npy`.
``` bash
  The files in base_dir shoule be organized as:
  ├── SEN12MS_CR                                      # This is your base dir name
  │   ├── ROIs1868_summer_s1                          # Subdir should be named as {seasons.value}_{sensor.value}
  │   │   ├── s1_102                                  # There should be named as  {sensor.value}_{scene_id}
  │   │   │   │   ├── ROIs1868_summer_s1_102_p100.tif # Named as  {seasons.value}_{sensor.value}_{scene_id}_{patch_id}.tif
  │   │   │   │   ├── ...
  │   │   ├── ...
  │   ├── ...
```
The command to execute the preprocessing script is as follows.
```bash
python preprocess.py -i <your origin dataset dir path> -o <your processed dataset dir path>
```

## Dataset split
The dataset consists of a total of 175 non-overlapping ROI regions. The dataset is split according to ROIs, with 10 ROIs as the test set, 10 ROIs as the validation set, and 155 ROIs as the training set.

We give a dataset split scheme and save it in a YAML file. You can get the details of the dataset split in `split.yaml`.

You can execute the following command if you want to resplit the dataset.
You will get a new dataset split file: `split.yaml`
```
cd experiment
export PYTHONPATH=$PWD
python tools/split.py --seed <your random seed> --dataset <your dataset path>
```
## Model training
We used wandb as an experimental monitoring and management scheme during model training. The `config-default.yaml` file was used for both the wandb configuration and the Trainer configuration.
1. config `config-default.yaml`
eg:
```yaml
model:
  value: 'TSOCR_V1'
epochs:
  value: 800
batch_size:
  value: 6
loss_fn:
  value: 'CharbonnierLoss'
save_dir:
  value: './runs'
lr:
  value: 0.0002
min_lr:
  value: 0.000001
validate_every:
  value: 5
dataset_file_extension:
  value: 'npy'
dataset:
  value: '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data_v2/PROCESSED_SEN12MS_CR/'
seed:
  value: 19990301
split_file_path:
  value: './split.yaml'
```

2. install warmup scheduler
```bash
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

We support both dataset-parallel training and distributed dataset-parallel training schemes in PyTorch.

3. use dataset-parallel for training

```bash
export CUDA_VISIBLE_DEVICES='0,1,2'
python train.py
```

4. use distributed-dataset-parallel for training

```
sh train.sh
```


