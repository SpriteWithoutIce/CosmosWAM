import torch


class ToTensor:
    """Convert uint8 images [0,255] to float [0,1]"""
    def __call__(self, image):
        if image.dtype == torch.uint8:
            return image.float() / 255.0
        return image
