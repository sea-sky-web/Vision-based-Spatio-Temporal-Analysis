import torchvision.transforms as T


def build_transforms(img_size=(256, 256)):
    """Build the image augmentation pipeline.

    We follow MVDet's practice by injecting moderate color jitter and
    ImageNet normalization to keep the encoder's activation statistics
    stable during training.
    """

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return T.Compose([
        T.Resize(img_size),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
        T.ToTensor(),
        normalize,
    ])