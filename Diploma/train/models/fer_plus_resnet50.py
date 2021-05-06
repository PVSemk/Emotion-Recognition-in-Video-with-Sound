import os
import six
import sys

import torch.nn as nn
from .resnet50 import Head, GRU_Head
import torch


class ResNetEmotionModel(nn.Module):
    def __init__(self, cfg, hidden_size=256):
        super(ResNetEmotionModel, self).__init__()
        self.cfg = cfg
        self.backbone = self.load_model()
        last_layer_name, last_module = list(self.backbone.named_modules())[-1]
        try:
            self.in_channels, self.out_channels = last_module.in_features, last_module.out_features
        except:
            self.in_channels, self.out_channels = last_module.in_channels, last_module.out_channels
        setattr(self.backbone, '{}'.format(last_layer_name), nn.Identity())  # the second last layer has 512 dimensions
        setattr(self, 'output_feature_dim', self.in_channels)

        pool_layer_name, pool_layer = list(self.backbone.named_modules())[-2]
        setattr(self.backbone, '{}'.format(pool_layer_name), nn.AdaptiveAvgPool2d((1, 1)))

        self.val_fc = Head(self.in_channels, hidden_size, self.cfg.digitize_number)
        self.arousal_fc = Head(self.in_channels, hidden_size, self.cfg.digitize_number)
        # self.val_fc = nn.Linear(in_channels, self.cfg.digitize_number)
        # self.arousal_fc = nn.Linear(in_channels, self.cfg.digitize_number)

    def forward(self, input, audio=None):
        features = self.backbone(input, audio).squeeze(-1).squeeze(-1) if audio is not None else self.backbone(input).squeeze(-1).squeeze(-1)
        val_pred = self.val_fc(features)
        arousal_pred = self.arousal_fc(features)
        return {'val_pred': val_pred, 'arousal_pred': arousal_pred}

    def load_model(self):
        """Load imoprted PyTorch model by name

        Args:
            model_name (str): the name of the model to be loaded

        Return:
            nn.Module: the loaded network
        """
        model_def_path = os.path.join(self.cfg.model_dir, self.cfg.model_name + '.py' if "audio_path" not in self.cfg else self.cfg.model_name + "_audio.py")
        weights_path = os.path.join(self.cfg.model_dir, self.cfg.model_name + '.pth')
        mod = self.load_module_2or3(model_def_path)
        func = getattr(mod, self.cfg.model_name)
        net = func(weights_path=weights_path)
        return net

    def load_module_2or3(self, model_def_path):
        """Load model definition module in a manner that is compatible with
        both Python2 and Python3

        Args:
            model_name: The name of the model to be loaded
            model_def_path: The filepath of the module containing the definition

        Return:
            The loaded python module."""
        if six.PY3:
            import importlib.util
            spec = importlib.util.spec_from_file_location(self.cfg.model_name, model_def_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            import importlib
            dirname = os.path.dirname(model_def_path)
            sys.path.insert(0, dirname)
            module_name = os.path.splitext(os.path.basename(model_def_path))[0]
            mod = importlib.import_module(module_name)
        return mod


class ResNetEmotionModelGRU(ResNetEmotionModel):
    def __init__(self, cfg, hidden_size=128):
        super().__init__(cfg, hidden_size)
        self.val_fc = GRU_Head(self.in_channels, hidden_size, self.cfg.digitize_number)
        self.arousal_fc = GRU_Head(self.in_channels, hidden_size, self.cfg.digitize_number)
        self.seq_len = self.cfg.seq_len

    def forward(self, input, audio=None):
        orig_shape = input.shape
        B, S, C, H, W = input.shape
        input = input.view(B*S, C, H, W)
        features = self.backbone(input, audio).squeeze(-1).squeeze(-1) if audio is not None else self.backbone(input).squeeze(-1).squeeze(-1)
        features = features.view(orig_shape[0], self.seq_len, -1)
        val_pred = self.val_fc(features)
        arousal_pred = self.arousal_fc(features)
        return {'val_pred': val_pred, 'arousal_pred': arousal_pred}


def create_model(cfg):
    if not cfg.rnn:
        return ResNetEmotionModel(cfg)
    else:
        return ResNetEmotionModelGRU(cfg)
