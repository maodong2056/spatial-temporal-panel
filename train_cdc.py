import os
import shutil
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision

from torch.autograd import Variable

from configuration import TRAIN
from inputs import PanelDataset, collate_fn

from cdc_net import CDCNet
from utils.layers import initialize_weights
from utils.easy_visdom import EasyVisdom
from utils.metric import accuracy, nll_loss

parser = argparse.ArgumentParser(description='PyTorch CDC Training')
parser.add_argument('-g', '--cuda-device', type=int, default=0,
                    help='choose which gpu to use (default: 0)')
parser.add_argument('-l', '--learning-rate', default=3e-7, type=float,
                    help='learning rate (default: 3e-7)')
parser.add_argument('-r', '--resume-path', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--experiment-name', default='untitled', type=str,
                    help='name of the experiment (default: untitled)')
parser.add_argument('--pretrained-path', default='', type=str,
                    help='path to the pre-trained model (default: none)')

args = parser.parse_args()

checkpoint_dir = TRAIN.checkpoint_root / args.experiment_name
result_dir = TRAIN.result_root / args.experiment_name
[d.mkdir(exist_ok=True) for d in [checkpoint_dir, result_dir]]


def main():
    best_loss = float('inf')  # best test loss

    # GPU (Default) configuration
    # ==========================================================================
    assert torch.cuda.is_available(), 'Use GPUs default, check cuda available'

    torch.cuda.manual_seed(TRAIN.seed)
    torch.cuda.set_device(args.cuda_device)
    print('===> Current GPU device is', torch.cuda.current_device())

    # Dataset loader
    # ==========================================================================
    print('==> Preparing data..')
    train_compress = TRAIN.compress
    test_compress = TRAIN.compress // 5

    trainset = PanelDataset(list_root=TRAIN.list_root, stage='train', compress=train_compress)
    testset = PanelDataset(list_root=TRAIN.list_root, stage='test', compress=test_compress)
    train_data_loader = data.DataLoader(trainset, batch_size=TRAIN.batch_size, shuffle=True,
                                        num_workers=TRAIN.num_workers, pin_memory=True, collate_fn=collate_fn)
    test_data_loader = data.DataLoader(testset, batch_size=TRAIN.batch_size, shuffle=False,
                                       num_workers=TRAIN.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Models
    # ==========================================================================
    model = CDCNet(num_classes=12 + 1).cuda()
    start_epoch = 1
    total_epoch = int(TRAIN.expect_epoch * train_compress)
    print('===> Building {}...'.format(args.experiment_name))
    print('---> Batch size: {}\tTotal epoch: {}\t'.format(TRAIN.batch_size, total_epoch))

    # Weights
    # ==========================================================================
    if args.resume_path:
        if os.path.isfile(args.resume_path):
            print("===> Loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("---> Loaded checkpoint at epoch: {}".format(start_epoch))
        else:
            print("---> No checkpoint found at '{}'".format(args.resume_path))
    else:
        print('===> Training from scratch at epoch: {}'.format(1))
        model.apply(initialize_weights)

    # Optimizer
    # ==========================================================================
    # 'features', 'cdc_6*', 'cdc_7*'
    base_parameters = [parms for name, parms in model.named_parameters() if not name.startswith('cdc_8')]
    classifier_parameters = [parms for name, parms in model.named_parameters() if name.startswith('cdc_8')]
    optimizer = optim.SGD([
        {'params': base_parameters},
        {'params': classifier_parameters, 'lr': args.learning_rate * 0.1}
    ], lr=args.learning_rate, momentum=0.9, weight_decay=0.005)

    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('---> Initial learning rate: {:.0e} '.format(args.learning_rate))

    # Visdom
    # ==========================================================================
    ev = EasyVisdom(from_scratch=not bool(args.resume_path),
                    start_i=start_epoch,
                    total_i=total_epoch,
                    mode=['train', 'test'],
                    stats=['loss', 'acc'],
                    results_dir=result_dir,
                    env=args.experiment_name)

    def train():
        epoch_loss, epoch_acc = np.zeros(2)

        # Sets the module in training mode, only on modules such as Dropout or BatchNorm.
        model.train()

        stamp = time.time()
        for partial_epoch, (volume, label) in enumerate(train_data_loader, 1):
            volume_var = Variable(volume).float().cuda()
            label_var = Variable(label).cuda()

            out = model(volume_var)

            loss = nll_loss(out, label_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(out, label_var)
            epoch_loss += loss.data[0]
            epoch_acc += acc.data[0]

        consume = time.time() - stamp
        avg_loss, avg_acc = np.array([epoch_loss, epoch_acc]) / partial_epoch
        print('===> Training epoch: {:.2f}/{}\t'.format(i / train_compress, TRAIN.expect_epoch),
              'Loss: {:.5f} | Accuracy: {:.5f}'.format(avg_loss, avg_acc),
              ' |  Elapsed: {:.3f}s / batch({})'.format(consume / len(train_data_loader), TRAIN.batch_size))

        return avg_loss, avg_acc, volume

    def test():
        epoch_loss, epoch_acc = np.zeros(2)

        # Sets the module in training mode, only on modules such as Dropout or BatchNorm.
        model.eval()

        for partial_epoch, (volume, label) in enumerate(test_data_loader, 1):
            volume_var = Variable(volume, volatile=True).float().cuda()
            label_var = Variable(label, volatile=True).cuda()

            out = model(volume_var)

            loss = nll_loss(out, label_var)
            acc = accuracy(out, label_var)
            epoch_loss += loss.data[0]
            epoch_acc += acc.data[0]

        avg_loss, avg_acc = np.array([epoch_loss, epoch_acc]) / partial_epoch
        if i % TRAIN.val_interval == 0:
            print('{}\n===> Validation  | '.format('-' * 130),
                  'Loss: {:.5f} | Accuracy: {:.5f}'.format(avg_loss, avg_acc),
                  'Current time: {}\n{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), '-' * 130))
        return avg_loss, avg_acc, volume

    for i in range(start_epoch, total_epoch + 1):
        # MultiStepLR monitors: epoch
        # =================================================================================
        # optimizer = step_lr_scheduler(optimizer, i, milestones=[800, 1000, 1500],
        #                               init_lr=args.learning_rate, instant=(start_i == i))
        # ==================================================================================

        # `results` contains [loss, accuracy]
        *train_results, train_volume = train()
        *test_results, test_volume = test()

        # Visualize - scalar
        ev.vis_scalar(i, train_results, test_results)
        # Visualize - images
        train_images = train_volume[0].permute(1, 0, 2, 3)[:36]
        test_images = test_volume[0].permute(1, 0, 2, 3)[:36]
        train_grid = torchvision.utils.make_grid(train_images, 6)[[0]].numpy()
        test_grid = torchvision.utils.make_grid(test_images, 6)[[0]].numpy()

        ev.vis_images(i,
                      im_titles=['sample'],
                      show_interval=TRAIN.plot_interval,
                      train_images=train_grid,
                      val_images=test_grid)

        # Save checkpoints
        if i % TRAIN.save_interval == 0:
            cur_loss = test_results[-1]
            is_best = cur_loss < best_loss
            best_loss = min(cur_loss, best_loss)

            state = {
                'epoch'     : i,
                'state_dict': model.state_dict(),
                'best_loss' : best_loss,
            }
            savename = str(checkpoint_dir / '{}_{}.pth'.format(args.experiment_name, i))
            bestname = str(checkpoint_dir / '{}_best.pth'.format(args.experiment_name))
            torch.save(state, savename)
            if is_best:
                shutil.copyfile(savename, bestname)
            np.save(str(result_dir / 'results_dict.npy'), ev.results_dict)


if __name__ == '__main__':
    main()
