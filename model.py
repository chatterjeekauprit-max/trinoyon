"""Small custom anchor-free grid detector (not YOLO). Requires PyTorch."""
import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(c_out), nn.SiLU(inplace=True))


class GridDetector(nn.Module):
    """Single-object-per-cell detector output channels: objectness, xywh, classes."""
    def __init__(self, num_classes: int, grid_size: int = 10):
        super().__init__()
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.grid_size = grid_size
        self.features = nn.Sequential(ConvBlock(3, 16, 2), ConvBlock(16, 24, 2),
            ConvBlock(24, 40, 2), ConvBlock(40, 64, 2), ConvBlock(64, 96, 2))
        self.head = nn.Conv2d(96, 5 + num_classes, 1)

    def forward(self, x):
        out = self.head(self.features(x))
        if out.shape[-2:] != (self.grid_size, self.grid_size):
            out = torch.nn.functional.interpolate(out, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        return out.permute(0, 2, 3, 1)


def encode(boxes, classes, grid_size, image_size):
    target = torch.zeros(grid_size, grid_size, 5 + len(classes))
    for box in boxes:
        cid, x, y, w, h = box
        cx, cy = x + w/2, y + h/2
        gx, gy = min(grid_size-1, int(cx/image_size*grid_size)), min(grid_size-1, int(cy/image_size*grid_size))
        target[gy, gx, 0] = 1
        target[gy, gx, 1:5] = torch.tensor([cx/image_size, cy/image_size, w/image_size, h/image_size])
        target[gy, gx, 5 + int(cid)] = 1
    return target


def decode(pred, class_names, width, height, threshold):
    """Decode one image's grid tensor to source pixel coordinates."""
    result = []
    grid = pred.shape[0]
    for gy in range(grid):
        for gx in range(grid):
            row = pred[gy, gx]
            obj = row[0].sigmoid().item()
            probs = row[5:].softmax(dim=0)
            score, cid = probs.max(dim=0)
            conf = obj * score.item()
            if conf < threshold:
                continue
            cx, cy, bw, bh = row[1:5].sigmoid().tolist()
            x, y = (cx-bw/2)*width, (cy-bh/2)*height
            result.append((int(cid), conf, max(0,x), max(0,y), min(width-x,bw*width), min(height-y,bh*height)))
    return result
