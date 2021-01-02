import torch
import os
import argparse

def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]

    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def print_args(args):
    message = ''
    message += '----------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        message += f'{str(k):>25}: {str(v):<30}\n'
    message += '----------------- End -------------------'
    return message

def save_checkpoint(model, optimizer, epoch, step, filepath):
    checkpoint_filename = f"checkpoint_{epoch:03d}.pth"
    checkpoint_path = os.path.join(filepath, checkpoint_filename)

    print("Saving at epoch {} to {}".format(epoch, checkpoint_path))

    checkpoint = {'epoch': epoch,
                  'step': step,
                  'random_rng_state': torch.random.get_rng_state(),
                  'model': model.state_dict(),
                  'optimizer':optimizer.state_dict()}
    
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, filepath):
    print("Loading models and optimizers from {}".format(filepath))

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch'], checkpoint['step'] 

def get_last_checkpoint_filename(logging_path):
    indices = []
    for root, dirs, files in os.walk(logging_path, topdown=False):
        for name in files:
            filename, extension = os.path.splitext(name)
            if extension == 'pth':
                indices.append(int(filename.split('_')[1]))
    
    return os.path.join(logging_path, f"checkpoint_{max(indices):03d}.pth")