import torchvision.transforms as T


def build_transforms(img_size=(256, 256)):
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
    ])