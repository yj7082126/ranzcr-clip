import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from dataset import ImageDataset
from model import Detector
from util import set_requires_grad, save_checkpoint, load_checkpoint, get_last_checkpoint_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RANZCR Arguments")

    dataset = parser.add_argument_group("dataset parameters")
    dataset.add_argument("--file-loc", default="data/train_new.csv", type=str,
            help="Path to training csv file")
    dataset.add_argument("--image-size", default=380, type=int,
            help="Output Image size")
    dataset.add_argument("--shuffle", default=True, type=bool,
            help="To not shuffle. ")
    dataset.add_argument("--do-transform", default=True, type=bool,
            help="To use image transform")

    training = parser.add_argument_group("training parameters")
    training.add_argument('--epochs', type=int, default=30,
                          help='Number of total epochs to run')
    training.add_argument('--batch-size', type=int, default=8,
                          help='Size of batch')
    training.add_argument('--epochs-per-checkpoint', type=int, default=1,
                          help='Number of epochs per checkpoint')
    training.add_argument('--resume-from-last', action='store_true',
                          help='Resumes training from the last checkpoint')
    training.add_argument('--dropout-rate', type=float, default=0.2,
                          help='Dropout rate for model')                      
    training.add_argument('--lr', type=float, default=1e-4,
                          help='Learning rate for Adam optimizer')
    training.add_argument('--min-lr', type=float, default=2e-6,
                          help='Minimal learning rate for scheduler')                          
    training.add_argument('--logging-path', type=str, default='logs/1e_4',
                          help='Logging path to log plots, losses and features')
    training.add_argument('--checkpoint-path', type=str, default='',
                          help='Checkpoint path to load a specified checkpoint')
    training.add_argument('--steps-per-logging', type=int, default=10,
                          help='Number of steps per logging')

    args = parser.parse_args()

    train_dataset = ImageDataset("train", args.image_size, args.file_loc, 
                                args.shuffle, fold=0, do_transform=args.do_transform)
    train_loader = DataLoader(train_dataset, num_workers=0, shuffle=True, 
        batch_size=args.batch_size, pin_memory=False, drop_last = True)
    val_dataset = ImageDataset("valid", args.image_size, args.file_loc, 
                                args.shuffle, fold=0, do_transform=False)
    val_loader = DataLoader(val_dataset, num_workers=0, shuffle=True, 
        batch_size=args.batch_size, pin_memory=False, drop_last = True)

    model = Detector(args.dropout_rate).cuda()
    set_requires_grad([model.feature_extractor], False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = args.epochs, eta_min = args.min_lr, last_epoch=-1
    )
    bce_loss = torch.nn.BCEWithLogitsLoss()

    epoch = 0 
    step = 0
    writer = SummaryWriter(args.logging_path)

    if args.resume_from_last:
        args.checkpoint_path = get_last_checkpoint_filename(args.logging_path)

    if args.checkpoint_path != "":
        epoch, step = load_checkpoint(model, optimizer, args.checkpoint_path)
    else:
        print("Start training from scratch.")

    for epoch in range(epoch, args.epochs):

        # Train
        model.train()
        set_requires_grad([model.feature_extractor], False)

        pbar = tqdm(enumerate(train_loader), 
            total=len(train_loader), desc=f"Epoch {epoch}", ncols=0)
        for i, sample in pbar:
            images = sample["image"].cuda()
            labels = sample["label"].cuda()
            optimizer.zero_grad()

            y = model(images)
            preds = y.sigmoid() > 0.5
            loss = bce_loss(y.squeeze(), labels)
            acc  = torch.sum(labels == preds).float() / (labels.size()[0] * labels.size()[1])

            loss.backward()
            optimizer.step()

            if (i + 1) % args.steps_per_logging == 0:
                writer.add_scalar('Loss/train', loss, step)
                pbar.set_postfix({
                    "lr"   : scheduler.get_last_lr(), 
                    "loss" : loss.item(), 
                    "acc"  : acc.item()})
        pbar.close()

        # Eval
        model.eval()

        pbar = tqdm(val_loader, 
            total=len(val_loader), desc=f"Epoch {epoch}", ncols=0)
        
        result_loss = []
        result_acc  = []
        with torch.no_grad():
            for sample in pbar:
                images = sample["image"].cuda()
                labels = sample["label"].cuda()
                img_name = sample["img_name"]

                y = model(images)
                preds = y.sigmoid() > 0.5
                loss = bce_loss(y.squeeze(), labels)
                acc  = torch.sum(labels == preds).float() / (labels.size()[0] * labels.size()[1])
                
                result_loss.append(loss.item())
                result_acc.append(acc.item())

        writer.add_scalar('Loss/valid', loss, epoch+1)
        writer.add_scalar('Acc/valid', acc, epoch+1)
        pbar.close()

        print(f"Epoch {epoch+1} : Loss {loss.item():.4f}, Acc {acc.item():.4f}")
        scheduler.step()

        if (epoch+1) % args.epochs_per_checkpoint == 0:
            save_checkpoint(model, optimizer, epoch, step, args.logging_path)