import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(config):
    transforms = [
        # A.ShiftScaleRotate(shift_limit=0.13, scale_limit=0.2, border_mode=0, value=[0, 0, 0]),
        A.Resize(config.height, config.width, interpolation=2),
        # A.HueSaturationValue(),
        # A.RandomBrightnessContrast(0.4, 0.4),
        # A.RandomGamma(),
        # A.HorizontalFlip(),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(transforms)


def val_transforms(config):
    return A.Compose([
        A.Resize(config.height, config.width, interpolation=2),
        A.Normalize(),
        ToTensorV2()])
