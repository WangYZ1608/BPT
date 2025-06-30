import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import Dataset
import timm, Models
from timm.utils import ModelEmaV2
from timm.models.layers import trunc_normal_
from utils import colorstr, Save_Checkpoint, adjust_learning_rate, param_groups_weight_decay

import numpy as np
from pathlib import Path
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        print("Use GPU: {} for training".format(args.gpu))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def create_dataset(args):
    train_dataset, val_dataset, num_class = Dataset.__dict__[args.dataset]()
    
    args.batch_size = int(args.batch_size / args.world_size)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        raise ValueError("Distributed init error.")
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            pin_memory=True,
                            sampler=val_sampler,
                            drop_last=False)
    
    return train_loader, val_loader, num_class, train_sampler


def create_model(args, num_class):
    if args.tuning_type in ["prompt"]:
        vpt = ['vpt_vit_b', 'vpt_vit_l', 'vpt_vit_h']
        vpt_deep = ['vpt_deep_vit_b', 'vpt_deep_vit_l', 'vpt_deep_vit_h']
        assert args.model_name in vpt + vpt_deep

        pretrain_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        pretrain_ckpt = pretrain_model.state_dict()

        model = Models.__dict__[args.model_name](
            drop_path_rate=args.drop_path,
            num_prompts=args.num_prompts,
            channels=args.channels,
            )
        
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in pretrain_ckpt.items() if k in model_dict and 'head' not in k}
        model_dict.update(state_dict)
        msg = model.load_state_dict(model_dict)
        print(msg)

        requires_grad = []
        for name, param in model.named_parameters():
            if name.split('.')[0] not in ['head', 'random_vectors', 'conv1x1']:
                param.requires_grad = False
            else:
                requires_grad.append(name)
        print("requires_grad: ", sorted(requires_grad))
    
    # modify the output dimension (num classes) according to the task
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features=in_features, out_features=num_class)

    # init task head weights.
    # * timm: std=0.02, MAE: std=2e-5
    trunc_normal_(model.head.weight, std=.02)
    nn.init.constant_(model.head.bias, 0)

    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    cudnn.benchmark = True

    device = torch.device(args.device)

    # data loaders
    train_loader, val_loader, num_class, train_sampler = create_dataset(args=args)

    # create model
    model = create_model(args=args, num_class=num_class)
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'head' not in n)
    print('number of params (M): %.4f' % (n_parameters / 1.e6))      

    param_groups = param_groups_weight_decay(model=model_without_ddp, weight_decay=args.weight_decay, weight_decay_head=args.wd_head)
    optimizer = torch.optim.AdamW(params=param_groups,
                                  lr=args.lr)
    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # file path
    if dist.get_rank() == 0:
        # weights
        save_dir = Path(args.save_dir)
        weights = save_dir / 'weights'
        weights.mkdir(parents=True, exist_ok=True)
        last = weights / 'last'
        best = weights / 'best'

        # acc,loss
        acc_loss = save_dir / 'acc_loss'
        acc_loss.mkdir(parents=True, exist_ok=True)
        train_acc_savepath = acc_loss / 'train_acc.npy'
        train_loss_savepath = acc_loss / 'train_loss.npy'
        val_acc_savepath = acc_loss / 'val_acc.npy'
        val_loss_savepath = acc_loss / 'val_loss.npy'

        # tensorboard
        logdir = save_dir / 'logs'
        logdir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(logdir, flush_secs=120)

        # model
        model_file = str(save_dir / 'model.txt')
        with open(model_file, "a") as f:
            print(model_without_ddp, file=f)
            print(args, file=f)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
           
        args.start_epoch = checkpoint['epoch']
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = torch.tensor(checkpoint['best_acc'])
        if args.gpu is not None:
            # best_acc may be from a checkpoint from a different GPU
            best_acc = best_acc.to(args.gpu)

        train_acc = checkpoint['train_acc']
        train_loss = checkpoint['train_loss']
        test_acc = checkpoint['test_acc']
        test_loss = checkpoint['test_loss']

        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])

        print(colorstr('green', 'Resuming training from {} epoch'.format(args.start_epoch)))
    else:
        best_acc = 0
        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []
    
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_epoch_loss, train_acc1 = train(model=model,
                                             train_loader=train_loader,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             args=args,
                                             epoch=epoch,
                                             model_ema=model_ema)
        
        val_epoch_loss, val_acc = validate(model=model,
                                           val_loader=val_loader,
                                           criterion=criterion,
                                           args=args)
        
        s = "Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, lr: {:.1e}".format(
            train_epoch_loss, train_acc1, val_epoch_loss, val_acc, optimizer.param_groups[0]['lr'])
        print(colorstr('green', s))

        if dist.get_rank() == 0:
            # save acc,loss
            train_loss.append(train_epoch_loss)
            train_acc.append(train_acc1)
            test_loss.append(val_epoch_loss)
            test_acc.append(val_acc)

            # save model
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            state = {
                'epoch': epoch + 1,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss,
            }
            if model_ema:
                state["model_ema"] = model_ema.state_dict()
            
            last_path = last / 'epoch_{}_loss_{:.4f}_acc_{:.3f}'.format(
                epoch + 1, val_epoch_loss, val_acc)
            best_path = best / 'epoch_{}_acc_{:.4f}'.format(
                epoch + 1, best_acc)
            Save_Checkpoint(state, last, last_path, best, best_path, is_best)
            
            summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
            summary_writer.add_scalar('train_acc', train_acc1, epoch)
            summary_writer.add_scalar('val_loss', val_epoch_loss, epoch)
            summary_writer.add_scalar('val_acc', val_acc, epoch)
    
    if dist.get_rank() == 0:
        summary_writer.close()
        if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
            np.save(train_acc_savepath, train_acc)
            np.save(train_loss_savepath, train_loss)
            np.save(val_acc_savepath, test_acc)
            np.save(val_loss_savepath, test_loss)


def train(model, train_loader, optimizer, criterion, args, epoch, model_ema):
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    # Model on train mode
    model.train()
    step_per_epoch = len(train_loader)
    for step, (images, labels) in enumerate(train_loader):
        torch.cuda.synchronize()
        start = time.time()

        adjust_learning_rate(optimizer, step / step_per_epoch + epoch, args)

        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
        
        # compute output
        logits = model(images)
        loss = criterion(logits, labels)

        # measure accuracy and record loss
        acc1 = accuracy(logits, labels, topk=(1, ))

        train_loss.update(loss.item(), images.size(0))
        train_acc.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time()-start), train_loss.val, train_acc.val)
        print(s1+s2, end='', flush=True)

    print()
    return train_loss.avg, train_acc.avg


def validate(model, val_loader, criterion, args):
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    # model to evaluate mode
    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            if args.gpu is not None and torch.cuda.is_available():
                 images = images.cuda(args.gpu, non_blocking=True)
                 labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            logits = model(images)
            loss = criterion(logits, labels)

            # measure accuracy and record loss
            acc1 = accuracy(logits, labels, topk=(1, ))

            # Average loss and accuracy across processes
            if args.distributed:
                loss = reduce_tensor(loss, args)
                acc1 = reduce_tensor(acc1[0], args)
            
            val_loss.update(loss.item(), images.size(0))
            val_acc.update(acc1[0].item(), images.size(0))
    
    return val_loss.avg, val_acc.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='PyTorch Training for visual tuning.')
    # model parameters
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224_in21k", help="architecture")
    parser.add_argument("--drop_path", type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--model_ema', action='store_true')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999)

    # * prompt
    parser.add_argument('--num_prompts', type=int, default=100, help="number of prompt tokens.")
    parser.add_argument('--prompt_deep', action='store_true', help="prompt type: shallow or deep")
    # parser.set_defaults(prompt_deep=True)
    parser.add_argument('--channels', type=int, default=16, help="conv channels for prompt positions")

    # optimizer parameters
    dataset_name = sorted(name for name in Dataset.__dict__
                          if not name.startswith("__") and callable(Dataset.__dict__[name]))
    parser.add_argument("--dataset", type=str, default='CUB200', choices=dataset_name)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--wd_head", type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    # tuning parameters
    parser.add_argument("--tuning_type", type=str, default="fine-tuning", choices=["fine-tuning", "linear", "prompt"])

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    parser.add_argument("--resume", type=str, help="ckpt's path to resume most recent training")
    parser.add_argument("--save_dir", type=str, default="./run", help="save path, eg, acc_loss, weights, tensorboard, and so on")
    args = parser.parse_args()

    print(colorstr('green', 'Fine-tuning ' + args.model_name + ' on ' + args.dataset + ' ...'))
    main(args=args)