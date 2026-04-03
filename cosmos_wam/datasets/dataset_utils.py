import torch
import torch.nn.functional as F


class ResizeSmallestSideAspectPreserving:
    def __init__(self, args):
        self.img_w = args["img_w"]
        self.img_h = args["img_h"]

    def __call__(self, video):
        # video: [T, C, H, W]
        _, _, h, w = video.shape
        if h / w > self.img_h / self.img_w:
            new_h = self.img_h
            new_w = int(w * new_h / h)
        else:
            new_w = self.img_w
            new_h = int(h * new_w / w)
        return F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)


class CenterCrop:
    def __init__(self, args):
        self.img_w = args["img_w"]
        self.img_h = args["img_h"]

    def __call__(self, video):
        _, _, h, w = video.shape
        top = (h - self.img_h) // 2
        left = (w - self.img_w) // 2
        return video[:, :, top : top + self.img_h, left : left + self.img_w]


class Normalize:
    def __init__(self, args):
        self.mean = args["mean"]
        self.std = args["std"]

    def __call__(self, video):
        return (video - self.mean) / self.std
