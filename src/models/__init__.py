'''
https://github.com/facebookresearch/detr
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .ecotr_model import build


def build_model(args):
    return build(args)
