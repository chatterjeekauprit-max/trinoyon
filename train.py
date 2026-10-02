"""Train the custom grid detector from a user-provided JSONL dataset."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from visionx.data import JsonlDetectionDataset
from visionx.model import GridDetector


def loss_fn(pred, target):
    obj = target[..., 0] > 0
    loss_obj = nn.functional.binary_cross_entropy_with_logits(pred[...,0], target[...,0])
    if not obj.any():
        return loss_obj
    loss_box = nn.functional.smooth_l1_loss(pred[...,1:5][obj].sigmoid(), target[...,1:5][obj])
    loss_cls = nn.functional.binary_cross_entropy_with_logits(pred[...,5:][obj], target[...,5:][obj])
    return loss_obj + 4.0*loss_box + loss_cls


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True); ap.add_argument("--val")
    ap.add_argument("--classes",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--image-size",type=int,default=320)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args(); names=[x.strip() for x in Path(args.classes).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not names: ap.error("classes file must contain at least one class")
    ds=JsonlDetectionDataset(args.train,names,args.image_size); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True)
    model=GridDetector(len(names),ds.grid_size).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    for epoch in range(args.epochs):
        model.train(); total=0
        for images, targets in loader:
            images,targets=images.to(args.device),targets.to(args.device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(images),targets); loss.backward(); opt.step(); total+=loss.item()
        print(f"epoch {epoch+1}/{args.epochs} train_loss={total/max(1,len(loader)):.5f}")
    if args.val:
        val=JsonlDetectionDataset(args.val,names,args.image_size,ds.grid_size)
        print(f"validation records available: {len(val)} (run visionx.evaluation.evaluate for metrics)")
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.cpu().state_dict(),"class_names":names,"image_size":args.image_size,
                "grid_size":ds.grid_size,"architecture":"visionx-grid-v1"},output)
    print(f"saved custom model checkpoint: {output}")

if __name__ == "__main__": main()
