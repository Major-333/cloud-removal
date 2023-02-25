import enum
import logging
from typing import List
import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

from models.dsen2cr import DSen2_CR
from models.mprnet import MPRNet
from models.restormer import Restormer
from models.TSOCR_V1 import TSOCR_V1
from models.TSOCR_V1m import TSOCR_V1m
from models.TSOCR_V2 import TSOCR_V2
from models.TSOCR_V2m import TSOCR_V2m
from models.TSOCR_V3 import TSOCR_V3
from models.test_model import TestModel
from models.cross_att_net import CrossAttNet, AblationNet1, AblationNet2, AblationNet3
from models.simulation_fusion_gan import SimulationNet


def _init_dsen2cr() -> nn.Module:
    model = DSen2_CR(in_channels=15, out_channels=13, num_layers=6, feature_dim=256)
    model = model.cuda()
    return model


def _init_mprnet() -> nn.Module:
    model = MPRNet()
    model = model.cuda()
    return model


def _init_test_model() -> nn.Module:
    model = TestModel()
    model = model.cuda()
    return model


def _init_restormer() -> nn.Module:
    model = Restormer()
    model = model.cuda()
    return model


def _init_TSOCR_V1_model() -> nn.Module:
    model = TSOCR_V1()
    model = model.cuda()
    return model


def _init_TSOCR_V2_model() -> nn.Module:
    model = TSOCR_V2()
    model = model.cuda()
    return model


def _init_TSOCR_V1m_model() -> nn.Module:
    model = TSOCR_V1m()
    model = model.cuda()
    return model


def _init_TSOCR_V2m_model() -> nn.Module:
    model = TSOCR_V2m()
    model = model.cuda()
    return model


def _init_TSOCR_V3_model() -> nn.Module:
    model = TSOCR_V3()
    model = model.cuda()
    return model

def _init_cross_att_net() -> nn.Module:
    model = CrossAttNet(13, 2)
    model = model.cuda()
    return model

def _init_ablation_net1() -> nn.Module:
    model = AblationNet1(13, 2)
    model = model.cuda()
    return model

def _init_ablation_net2() -> nn.Module:
    model = AblationNet2(13, 2)
    model = model.cuda()
    return model

def _init_ablation_net3() -> nn.Module:
    model = AblationNet3(13, 2)
    model = model.cuda()
    return model

def _init_simulation_net() -> nn.Module:
    model = SimulationNet(in_channels=2, out_channels=13)
    model = model.cuda()
    return model

MODEL_MAPPER = {
    'MPRNet': _init_mprnet,
    'Restormer': _init_restormer,
    'DSen2CR': _init_dsen2cr,
    'Test': _init_test_model,
    'TSOCR_V0': _init_restormer,
    'TSOCR_V0.5': _init_test_model,
    'TSOCR_V1': _init_TSOCR_V1_model,
    'TSOCR_V1m': _init_TSOCR_V1m_model,
    'TSOCR_V2': _init_TSOCR_V2_model,
    'TSOCR_V2m': _init_TSOCR_V2m_model,
    'TSOCR_V3': _init_TSOCR_V3_model,
    'CrossAttNet': _init_cross_att_net,
    'AblationNet1': _init_ablation_net1,
    'AblationNet2': _init_ablation_net2,
    'AblationNet3': _init_ablation_net3,
    'SimulationNet': _init_simulation_net
}


def build_model_with_dp(model_name: str, gpu_list: List[int]) -> nn.Module:
    model = build_model(model_name)
    logging.info(f'===== using gpu:{gpu_list} =====')
    return DP(model, device_ids=gpu_list)

def build_distributed_model(model_name: str, gpu_id: int) -> nn.Module:
    model = build_model(model_name)
    logging.info(f'===== using gpu:{gpu_id} =====')
    return DDP(model, device_ids=[gpu_id])

def build_distributed_gan_model(model_name: str, gpu_id: int) -> nn.Module:
    model = build_model(model_name)
    logging.info(f'===== using gpu:{gpu_id} =====')
    # For GAN model, we must set "broadcast_buffers=False"
    return DDP(model, device_ids=[gpu_id], broadcast_buffers=False)

def build_pretrianed_model_with_dp(model_name: str, checkpoint_path: str, gpu_list: List[int]) -> nn.Module:
    model = build_pretrained_model(model_name, checkpoint_path)
    logging.info(f'===== using gpu:{gpu_list} =====')
    return DP(model, device_ids=gpu_list)

def build_distributed_pretrained_model(model_name: str, checkpoint_path: str, gpu_id: int) -> nn.Module:
    model = build_pretrained_model(model_name, checkpoint_path)
    logging.info(f'===== using gpu:{gpu_id} =====')
    return DDP(model, device_ids=[gpu_id])

def build_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_MAPPER:
        raise ValueError(f'model name:{model_name} hasn\'t been supported. choose one of: {MODEL_MAPPER.keys()}')
    model = MODEL_MAPPER[model_name]()
    logging.info(f'===== using model: {type(model)} =====')
    model = model.cuda()
    return model


def build_pretrained_model(model_name: str, checkpoint_path: str) -> nn.Module:
    model = build_model(model_name)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    return model
