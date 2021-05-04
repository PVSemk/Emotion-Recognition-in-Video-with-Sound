import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Head, self).__init__()
        # self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc_0 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x = self.bn0(x)
        # f0 = self.bn1(F.relu(self.fc_0(x)))
        # output = self.fc_1(f0)
        f0 = F.relu(self.fc_0(x))
        output = self.fc_1(f0)

        return output


class ResNetEmotionModel(nn.Module):
    def __init__(self, backbone, cfg, hidden_size=512):
        super(ResNetEmotionModel, self).__init__()
        self.backbone = backbone  #
        in_features = backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.cfg = cfg

        self.val_fc = Head(in_features, hidden_size, self.cfg.digitize_number)
        self.arousal_fc = Head(in_features, hidden_size, self.cfg.digitize_number)

    def forward(self, input):
        features = self.backbone(input)
        val_pred = (self.val_fc(features))
        arousal_pred = (self.arousal_fc(features))
        return {'val_pred': val_pred, 'arousal_pred': arousal_pred}


def create_model(cfg):
    model = models.resnet50(pretrained=True)
    return ResNetEmotionModel(model, cfg)