from __future__ import division

from models import *
from my_utils.utils import *
from my_utils.datasets import *
from my_utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

try:
    import moxing as mox
except:
    print('not use moxing')


def prepare_data_on_modelarts(args):
    """
    将OBS上的数据拷贝到ModelArts中
    """
    # 拷贝预训练参数文件

    # 默认使用ModelArts中的如下两个路径用于存储数据：
    # 0) /cache/model: 如果适用预训练模型，存储从OBS拷贝过来的预训练模型
    # 1）/cache/datasets: 存储从OBS拷贝过来的训练数据
    # 2）/cache/log: 存储训练日志和训练模型，并且在训练结束后，该目录下的内容会被全部拷贝到OBS
    if args.pretrained_weights:
        _, weights_name = os.path.split(args.pretrained_weights)
        mox.file.copy(args.pretrained_weights, os.path.join(args.local_data_root, 'model/'+weights_name))
        args.pretrained_weights = os.path.join(args.local_data_root, 'model/'+weights_name)
    if not (args.data_url.startswith('s3://') or args.data_url.startswith('obs://')):
        args.data_local = args.data_url
    else:
        args.data_local = os.path.join(args.local_data_root, 'datasets/trainval')
        if not os.path.exists(args.data_local):
            data_dir = os.path.join(args.local_data_root, 'datasets')
            mox.file.copy_parallel(args.data_url, data_dir)
            os.system('cd %s;unzip trainval.zip' % data_dir)  # 训练集已提前打包为trainval.zip
            if os.path.isdir(args.data_local):
                os.system('cd %s;rm trainval.zip' % data_dir)
                print('unzip trainval.zip success, args.data_local is', args.data_local)
            else:
                raise Exception('unzip trainval.zip Failed')
        else:
            print('args.data_local: %s is already exist, skip copy' % args.data_local)

    if not (args.train_url.startswith('s3://') or args.train_url.startswith('obs://')):
        args.train_local = args.train_url
    else:
        args.train_local = os.path.join(args.local_data_root, 'log/')
        if not os.path.exists(args.train_local):
            os.mkdir(args.train_local)

    return args


def gen_model_dir(args, model_best_path):
    current_dir = os.path.dirname(__file__)
    mox.file.copy_parallel(os.path.join(current_dir, 'deploy_scripts'),
                           os.path.join(args.train_url, 'model'))
    mox.file.copy_parallel(os.path.join(current_dir, 'my_utils'),
                           os.path.join(args.train_url, 'model/my_utils'))
    mox.file.copy(os.path.join(current_dir, 'config/yolov3-44.cfg'),
                  os.path.join(args.train_url, 'model/yolov3-44.cfg'))
    mox.file.copy(os.path.join(current_dir, 'config/train_classes.txt'),
                  os.path.join(args.train_url, 'model/train_classes.txt'))
    mox.file.copy(os.path.join(current_dir, 'config/classify_rule.json'),
                  os.path.join(args.train_url, 'model/classify_rule.json'))
    mox.file.copy(os.path.join(current_dir, 'models.py'),
                  os.path.join(args.train_url, 'model/models.py'))
    mox.file.copy(model_best_path,
                  os.path.join(args.train_url, 'model/models_best.pth'))
    print('gen_model_dir success, model dir is at', os.path.join(args.train_url, 'model'))


def freeze_body(model, freeze_body):
    # input: freeze_body.type = int, .choose = 0, 1, 2
    # return: modified model.parameters()
    # notes:
    #   0: do not freeze any layers
    #   1: freeze Darknet53 only
    #   2: freeze all but three detection layers
    #   three detection layers is [81, 93, 105], refer to https://blog.csdn.net/litt1e/article/details/88907542

    for name, value in model.named_parameters():
        value.requires_grad = True

    if freeze_body == 0:
        print('using original model without any freeze body')
    elif freeze_body == 1:
        print('using fitting model with backbone(Darknet53) frozen')
        for name, value in model.named_parameters():
            layers = int(name.split('.')[1])
            if layers < 74:
                value.requires_grad = False
    elif freeze_body == 2:
        print('using fitting model with all but three detection layers frozen')
        for name, value in model.named_parameters():
            layers = int(name.split('.')[1])
            if layers not in [81, 93, 105]:
                value.requires_grad = False
    else:
        print('Type error for freeze_body. Thus no layer is frozen')

    new_params = filter(lambda p: p.requires_grad, model.parameters())
    return new_params


def train(model, dataloader, optimizer, epoch, opt, device):
    model.train()
    start_time = time.time()
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + batch_i

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets)
        loss.backward()

        if batches_done % opt.gradient_accumulations:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

        #   Log progress
        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.max_epochs_2, batch_i, len(dataloader))
        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

            '''
            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{j+1}", metric)]
            tensorboard_log += [("loss", loss.item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)
            '''

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        epoch_batches_left = len(dataloader) - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)

        model.seen += imgs.size(0)


def valid(model, path, class_names, opt):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=path,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=opt.img_size,
        batch_size=32,
    )
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    # logger.list_of_scalars_summary(evaluation_metrics, epoch)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")
    return AP


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs_1', default=10, type=int, help='number of total epochs to run in stage one')
    parser.add_argument('--max_epochs_2', default=10, type=int, help='number of total epochs to run in total')
    parser.add_argument("--freeze_body", type=int, default=2, help="frozen specific layers for stage one")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate for stage two")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="PyTorch-YOLOv3-ModelArts/config/yolov3-44.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="PyTorch-YOLOv3-ModelArts/config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights, "
                                                                           "here set same to evaluation_interval")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument('--local_data_root', default='/cache/', type=str,
                        help='a directory used for transfer data between local path and OBS path')
    parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
    parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
    parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
    parser.add_argument('--train_local', default='', type=str, help='the training output results on local')
    parser.add_argument('--init_method', default='', type=str, help='the training output results on local')

    opt = parser.parse_args()
    print(opt)
    opt = prepare_data_on_modelarts(opt)

    # logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # store the name of model with best mAP
    model_best = {'mAP': 0, 'name': ''}

    # first stage training to get a relatively stable model
    optimizer_1 = torch.optim.Adam(freeze_body(model, opt.freeze_body), lr=1e-3)
    for epoch in range(opt.max_epochs_1):

        train(model, dataloader, optimizer_1, epoch, opt, device)

        if epoch % opt.evaluation_interval == 0:
            AP = valid(model, valid_path, class_names, opt)

            temp_model_name = f"ckpt_%d_%.2f.pth" % (epoch, 100 * AP.mean())
            ckpt_name = os.path.join(opt.train_local, temp_model_name)
            torch.save(model.state_dict(), ckpt_name)
            mox.file.copy_parallel(ckpt_name, os.path.join(opt.train_url, temp_model_name))

            if AP.mean() > model_best['mAP']:
                model_best['mAP'] = AP.mean()
                model_best['name'] = ckpt_name

    # second stage training to achieve higher mAP
    optimizer_2 = torch.optim.Adam(freeze_body(model, 0), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=10)
    for epoch in range(opt.max_epochs_1, opt.max_epochs_2):

        train(model, dataloader, optimizer_2, epoch, opt, device)

        if epoch % opt.evaluation_interval == 0:
            AP = valid(model, valid_path, class_names, opt)

            temp_model_name = f"ckpt_%d_%.2f.pth" % (epoch, 100 * AP.mean())
            ckpt_name = os.path.join(opt.train_local, temp_model_name)
            torch.save(model.state_dict(), ckpt_name)
            mox.file.copy_parallel(ckpt_name, os.path.join(opt.train_url, temp_model_name))

            if AP.mean() > model_best['mAP']:
                model_best['mAP'] = AP.mean()
                model_best['name'] = ckpt_name

        scheduler.step(epoch)

        print('The current learning rate is: ', scheduler.get_lr()[0])

    gen_model_dir(opt, model_best['name'])
