import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ImageDataset
from model import Detector
from util import set_requires_grad, save_checkpoint, load_checkpoint, get_last_checkpoint_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RANZCR Arguments")

    dataset = parser.add_argument_group("dataset parameters")
    dataset.add_argument("--file-loc", 
            default="data/train.csv", type=str,
            help="Path to training csv file")
    dataset.add_argument("--image-size", 
            default=512, type=int,
            help="Output Image size")
    dataset.add_argument("--shuffle", 
            default=True, type=bool,
            help="To not shuffle. ")
    dataset.add_argument("--do-transform", 
            default=True, type=bool,
            help="To use image transform")

    training = parser.add_argument_group("training parameters")
    training.add_argument('--epochs', type=int, default=6,
                          help='Number of total epochs to run')
    training.add_argument('--batch-size', type=int, default=8,
                          help='Size of batch')
    training.add_argument('--epochs-per-checkpoint', type=int, default=1,
                          help='Number of epochs per checkpoint')
    training.add_argument('--resume-from-last', action='store_true',
                          help='Resumes training from the last checkpoint')
    training.add_argument('--learning-rate', type=float, default=2e-5,
                          help='Learning rate for Adam optimizer')
    training.add_argument('--logging-path', type=str, default='logs/',
                          help='Logging path to log plots, losses and features')
    training.add_argument('--checkpoint-path', type=str, default='checkpoints',
                          help='Checkpoint path to load a specified checkpoint')
    training.add_argument('--steps-per-logging', type=int, default=100,
                          help='Number of steps per logging')

    args = parser.parse_args()

    train_dataset = ImageDataset(args.image_size, args.file_loc, 
                                args.shuffle, args.do_transform)
    train_loader = DataLoader(train_dataset, num_workers=0, shuffle=True, 
        batch_size=args.batch_size, pin_memory=False, drop_last = True)
    
    model = Detector(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        print('\nEpoch', epoch)
        if epoch % args.epochs_per_checkpoint == 0:
            save_checkpoint(model, optimizer, epoch, step, args.logging_path)

        for i, (images, labels) in enumerate(train_loader):
            # Data
            images = images.cuda() # fake 1, real 0
            labels = labels.cuda()

            # Model
            #set_requires_grad([model], True)
            model.train()
            set_requires_grad([model.feature_extractor], False)
            y = model(images)
            loss = bce_loss(y.squeeze(), labels)

            # Update
            loss.backward()
            optimizer.step()

            # Log
            writer.add_scalar('Loss/train', loss, step)
