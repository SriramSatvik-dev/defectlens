from torchvision import transforms
IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _build_transforms():
    """Returns the two transforms applied to the SAME resized image:
    one ImageNet-normalized (encoder input), one plain [0,1] (recon target)."""
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor = transforms.ToTensor()  # -> [0,1]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return resize, to_tensor, normalize