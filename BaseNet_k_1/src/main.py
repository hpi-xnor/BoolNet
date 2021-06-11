import math
import time
import pandas as pd
import argparse
import os
import numpy as np
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid 

from models.modules import MultiBConv, BConv, GhostSign, GhostBNSign
from models.boolnet import BasicBlock

from utils.transform import optim_params, input_transforms
from utils.KD_Loss import DistributionLoss
from utils.utils import automatic_gradient_clip
from utils.radam import RAdam
from torchtoolbox.tools import  mixup_data, mixup_criterion
from torchtoolbox.nn import LabelSmoothingLoss

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
    
parser = argparse.ArgumentParser(description="PyTorch BoolNet Training")
parser.add_argument("--dataset", default="CIFAR10", help="dataset to be used")
parser.add_argument("--model", default="boolnet18", help="backbone to be used")
parser.add_argument("--train_batch_size", default = 128, type = int, help="training batch")
parser.add_argument("--test_batch_size", default = 128, type = int, help="testing batch")
parser.add_argument("--save_model_freq", default=50, type=int, help="frequency to save model")
parser.add_argument("--print_freq", default=10, type=int, help="frequency to print state")
parser.add_argument("--cuda", default=True, type=bool, help="frequency to save model")
parser.add_argument("--parallel", default=0, type=int, help = "number of GPUs for parallel training")
parser.add_argument("--continue_training", action="store_true", help="search for checkpoint in the subfolder specified by `task` argument")
parser.add_argument("--pretrained_dir", default="checkpoints/ImageNet/boolnet18/boolnet18_student_lablesmooth_False_mixup_False_slices_1_best.pth", help="path to folder containing the model")
parser.add_argument("--teachers", default = "resnet34", type = str, help = "specify the teacher model used in distillation")
parser.add_argument("--teacher_dir", default="checkpoints/CIFAR10/resnet18/250_resnet18_teacher.pth", help="path to folder containing the model")
parser.add_argument("--lr", default = 1e-3, type = float, help="trainining learning rate")
parser.add_argument("--weight_decay", default = 0, type = float, help="training weight decay")
parser.add_argument('--num_workers', type = int, default = 4, help = "dataloader workers")
parser.add_argument('--dali_cpu', action = 'store_true', help = 'loading data using cpu')
parser.add_argument('--optimizer', type = str, default = 'Adam', help = "optimier to be used")
parser.add_argument('--epoch', type = int, default = -1, help = "training epochs" )
parser.add_argument('--start_epoch', type = int, default = 0, help = "training epochs" ) 
parser.add_argument('--gpu', type = int, default = 0, help = "gpu to be selected" ) 
parser.add_argument('--acceleration', type = bool, default = False, help = "enable FP16 acceleration" )
parser.add_argument('--lr_scheduler', type = str, default = "cosine", help = "tyep of lr scheduler")
parser.add_argument("--mode", type = str, default = "student", help = "student/teacher/distillation")
parser.add_argument("--self_distillation", action="store_true", help = "training with self distillation ")
parser.add_argument("--mixup", action="store_true", help = "training with data mixup")
parser.add_argument("--labelsmoothing", action="store_true", help = "training with labelsmoothing")
parser.add_argument("--max_slices", type = int, default = 8, help = "specify the number of max slices")
parser.add_argument("--summary_directory", type = str, default = "runs/" + time.strftime("/%Y-%m-%d-%H-%M/", time.localtime()),
                    help = "specify directory for tensorboard directories")
parser.add_argument("--imagenet_directory", type = str, default = "/mnt/imagenet/", help = "imagenet directory")
parser.add_argument("--update_interval", type = int, default = -1, help = "for printing and gradient accumulation")
parser.add_argument("--binary_downsample", action="store_true", help = "use binary 1x1 downsample conv")

global logger

def main():
    opt = parser.parse_args()

    logger = SummaryWriter("{}/lablesmooth_{}_mixup_{}_slices_{}/".format(opt.summary_directory, opt.labelsmoothing, opt.mixup, opt.max_slices))

    cuda = opt.cuda

    if cuda and not torch.cuda.is_available():

        raise Exception("No GPU found, please run without --cuda")

    seed = 1314

    torch.manual_seed(seed)

    if cuda:

        torch.cuda.manual_seed(seed)
        
        torch.cuda.set_device(opt.gpu)
        
    cudnn.benchmark = True
    if opt.dataset == "CIFAR10":
        opt.num_classes = 10
        opt.epoch = 255 if opt.epoch < 0 else opt.epoch
        opt.update_interval = 128 if opt.update_interval < 0 else opt.update_interval

    if opt.dataset == "CIFAR100":
        opt.num_classes = 100
        opt.epoch = 255 if opt.epoch < 0 else opt.epoch
        opt.update_interval = 128 if opt.update_interval < 0 else opt.update_interval

    if opt.dataset == "ImageNet":
        opt.num_classes = 1000
        opt.epoch = 60 if opt.epoch < 0 else opt.epoch
        opt.update_interval = 256 if opt.update_interval < 0 else opt.update_interval
    
    print("=======> Building {} dataset".format(opt.dataset))
    train_dataloader, test_dataloader = get_data(opt)
    
    print("=======> Building {} network".format(opt.model))

    assert opt.mode in {"student", "teacher", "distillation"}
    print("=======> Training mode {} for {} epochs".format(opt.mode, opt.epoch))

    if not opt.mode == "distillation":
      teacher = None
      network = get_model(opt, num_classes = opt.num_classes, mode = opt.mode, model = opt.model)

    else:
      network = get_model(opt, num_classes=opt.num_classes, mode="student", model=opt.model)
      teacher = []
      teacher_model_names = opt.teachers.split(",")
      for teacher_name in teacher_model_names:
        teacher.append(set_device(opt, get_model(opt, num_classes=opt.num_classes, mode="teacher", model=teacher_name), None))

      # load_state_dict(teacher, None, None, opt.teacher_dir)

      for model in teacher:
        for param in model.parameters():
          param.requires_grad is False
 
    print("=======> Building {} optimizer and {} lr_scheduler".format(opt.optimizer, opt.lr_scheduler))
    train_params = [{'params':[param for name, param in network.named_parameters()], "lr": opt.lr, "weight_decay": opt.weight_decay}]
    
    train_optimizer, train_lr_scheduler = get_optimizer(opt, train_params)
    
    if opt.continue_training:
      state_dict = load_state_dict(network, train_optimizer, train_lr_scheduler, opt.pretrained_dir)
      
    print("=======> Setting computation device")
    network = set_device(opt, network, train_optimizer)

    print("=======> Start training")
    Best_Top1 = 0 if not opt.continue_training else state_dict["best_acc"]
    temperature = 1 
    freq = 5
    running_best_acc = [] if not opt.continue_training else state_dict["running_best_acc"]
    running_train_prec1 = [] if not opt.continue_training else state_dict["running_train_prec1"]
    running_train_prec5 = [] if not opt.continue_training else state_dict["running_train_prec5"]
    running_train_loss = [] if not opt.continue_training else state_dict["running_train_loss"]

    running_test_prec1 = [] if not opt.continue_training else state_dict["running_test_prec1"]
    running_test_prec5 = [] if not opt.continue_training else state_dict["running_test_prec5"]
    running_test_loss = []if not opt.continue_training else state_dict["running_test_loss"]
    running_epochs = [] if not opt.continue_training else state_dict["running_epochs"]
    opt.start_epoch = opt.start_epoch if not opt.continue_training else state_dict["epoch"]


    for i in range(opt.start_epoch, opt.epoch, 1):
      
      avg_train_prec1, avg_train_prec5, avg_train_loss =  torch.zeros(1), torch.zeros(1), torch.zeros(1)
       
      if opt.dataset != "ImageNet":
        temperature = set_temperature_cifar(network, opt, i, freq = freq)
      
      else:
        temperature = 1.0
        
      avg_train_prec1, avg_train_prec5, avg_train_loss, inputs = Train(
                                                                  network, 
                                                                  teacher,
                                                                  train_dataloader, 
                                                                  train_optimizer, 
                                                                  opt, 
                                                                  i,
                                                                  temperature,
                                                                  logger,
                                                                  )
          
      avg_test_prec1, avg_test_prec5, avg_test_loss = Test(network, test_dataloader, opt)
      
      if opt.lr_scheduler in {"multisep", "cosine", "exponential"}:
        train_lr_scheduler.step()
          
      elif opt.lr_scheduler == "plateau":
        train_lr_scheduler.step(avg_test_prec1)

      running_epochs.append(i)
      running_best_acc.append(Best_Top1)
      
      running_train_prec1.append(avg_train_prec1.data.cpu().numpy())
      running_train_prec5.append(avg_train_prec5.data.cpu().numpy())
      running_train_loss.append(avg_train_loss.data.cpu().numpy())

      running_test_prec1.append(avg_test_prec1.data.cpu().numpy())
      running_test_prec5.append(avg_test_prec5.data.cpu().numpy())
      running_test_loss.append(avg_test_loss.data.cpu().numpy())

      dataframe = pd.DataFrame({'epochs': running_epochs, 'running_train_loss': running_train_loss, 'running_test_loss': running_test_loss, 'running_train_prec1': running_train_prec1, 'running_train_prec5': running_train_prec5, 'running_test_prec1':running_test_prec1, 'running_test_prec5':running_test_prec5})

      dataframe.to_csv("{}/{}_{}_lablesmooth_{}_mixup_{}_slices_{}_best.csv".format(opt.summary_directory, opt.model, opt.mode, opt.labelsmoothing, opt.mixup, opt.max_slices), index = False)
      
      logger.add_scalar('best_prec1', Best_Top1, i)          
      
      logger.add_scalar('train_loss', avg_train_loss.item(), i)
      logger.add_scalar('train_prec1', avg_train_prec1.item(), i)
      logger.add_scalar('train_prec5', avg_train_prec5.item(), i)
        
      logger.add_scalar('test_loss', avg_test_loss.item(), i)
      logger.add_scalar('test_prec1', avg_test_prec1.item(), i)
      logger.add_scalar('test_prec5', avg_test_prec5.item(), i)

      if avg_test_prec1 > Best_Top1:
          Best_Top1 = avg_test_prec1
          save_state_dict(
                  net = network.module if opt.parallel else network,
                  train_optimizer = train_optimizer,
                  train_lr_scheduler = train_lr_scheduler,
                  epoch = i,
                  opt = opt,
                  best_acc = Best_Top1,
                  running_epochs=running_epochs,
                  running_best_acc = running_best_acc,
                  running_train_prec1 = running_train_prec1,
                  running_train_prec5 = running_train_prec5,
                  running_train_loss = running_train_loss,
                  running_test_prec1 = running_test_prec1,
                  running_test_prec5 = running_test_prec5,
                  running_test_loss = running_test_loss
                  )
                  
    logger.close()
    print("======> Finish Trining")
            
def Train(network, teacher, dataloader, optimizer, opt, epochs, temperature, logger):
    
    avg_prec1 = torch.zeros(1)
    avg_prec5 = torch.zeros(1)
    avg_loss = torch.zeros(1)
    
    kl_loss_slice = 0
    ss_loss = 0
    self_surpervised_loss = 0
    self_distillation_loss = 0
    network.train()
    
    if teacher is not None:
      for model in teacher:
          model.eval()

    for iteration, batch in enumerate(dataloader, 1):
      if not opt.dataset == "ImageNet":
        for i in range(len(batch)):
          if batch[i] is not None:
            if opt.cuda:
              batch[i] = batch[i].cuda()
            else:
              batch[i] = batch[i].cpu()
        inputs, labels = Variable(batch[0]), Variable(batch[1])
        
      else:
        inputs = Variable(batch[0]['data'])
        labels = Variable(batch[0]['label'].squeeze().long().cuda())        
      
      if opt.mixup:
          inputs, labels_a, labels_b, lamb =  mixup_data(inputs, labels, 0.2)
      
      if opt.dataset == "ImageNet":
        default_epochs = 60
        default_frequency = 1000
        temperature = set_temperature_imagenet(network, opt, iteration+epochs*(dataloader._size//opt.train_batch_size),
                                               # freq=int(opt.epoch / default_epochs * default_frequency))
                                               freq=default_frequency)

      btic = time.time()
      outputs = network(inputs)
      
      outputs = outputs.squeeze()
      
      #self_distillation_loss = sum(self_distillation_loss).mean().squeeze()

      if teacher is not None:
        with torch.no_grad():
            outputs_teacher = torch.cat([F.softmax(teacher[i](inputs), dim=1).unsqueeze(-1) for i in range(len(teacher))], dim = -1).mean(-1)

        distillation_loss = DistributionLoss()(outputs, outputs_teacher)

        cross_entropy_loss = 0
      
      else:
        distillation_loss = 0
      
        if opt.mode == "pretrain":
          labels = torch.randint_like(labels, 0, 9).long()
        
        if opt.labelsmoothing:
          lossfunction = LabelSmoothingLoss(opt.num_classes, smoothing=0.1)
        
        else:
          lossfunction = nn.CrossEntropyLoss()
        
        if not opt.mixup:
          
            cross_entropy_loss = lossfunction(outputs, labels)
        
        else:
            cross_entropy_loss = mixup_criterion(lossfunction, outputs, labels_a, labels_b, lamb)

      
      loss = cross_entropy_loss + distillation_loss
      
      top1, top5 = accuracy(outputs, labels, (1,5))
      
      loss.backward()        
      
      if (iteration%(opt.update_interval//opt.train_batch_size))==0:

        torch.nn.utils.clip_grad_norm_(network.parameters(), 2.5)
        #automatic_gradient_clip(network)
        
        optimizer.step()
      
        optimizer.zero_grad()
       
      if iteration % (opt.print_freq*(opt.update_interval//opt.train_batch_size)) == 0 and iteration != 0:
            print("[# TRAIN # Epoch %d, Total Iterations %6d] Loss: %.4f  Distillation_Loss: %.4f Cross_Entropy_loss: %.4f LR: %.6f T: %.6f Speed: %4.1f samples/s" % (epochs + 1, (iteration + 1), loss.mean().item(), distillation_loss, cross_entropy_loss, optimizer.param_groups[0]['lr'], temperature, opt.train_batch_size/(time.time()-btic)))

      btic = time.time()
      avg_prec1 += (top1).cpu()
      avg_prec5 += (top5).cpu()
      avg_loss += (loss).cpu()

    avg_prec1 = (100 * avg_prec1)/(iteration*opt.train_batch_size)
    avg_prec5 = (100 * avg_prec5)/(iteration*opt.train_batch_size)
    avg_loss = avg_loss/(iteration)
    
    return avg_prec1, avg_prec5, avg_loss, inputs

def Test(network, test_dataloader, opt):
    
    avg_test_prec1 = torch.zeros(1)
    avg_test_prec5 = torch.zeros(1)
    avg_test_loss = torch.zeros(1)
    with torch.no_grad():
      for index, batch in enumerate(test_dataloader, 1):
          if not opt.dataset == "ImageNet":
            for i in range(len(batch)):
              if batch[i] is not None:
                if opt.cuda:
                  batch[i] = batch[i].cuda()
                else:
                  batch[i] = batch[i].cpu()
            inputs, labels = Variable(batch[0]), Variable(batch[1])
      
          else:
            inputs = Variable(batch[0]['data'])
            labels = Variable(batch[0]['label'].squeeze().long().cuda())        
      
          network.eval()
          
          outputs = network(inputs)
          
          test_prec1, test_prec5 = accuracy(outputs, labels, (1,5))
          #print(test_prec1)
          avg_test_prec1 += test_prec1.cpu()
          avg_test_prec5 += test_prec5.cpu()

          avg_test_loss += F.cross_entropy(outputs, labels).data.cpu()
    #print(avg_test_prec1)
    #print(index*opt.test_batch_size)      
    #avg_test_prec1 = (100 * avg_test_prec1).float()/len(test_dataloader.dataset)
    #avg_test_prec5 = (100 * avg_test_prec5).float()/len(test_dataloader.dataset)
    avg_test_prec1 = (100 * avg_test_prec1).float()/(index*opt.test_batch_size)
    avg_test_prec5 = (100 * avg_test_prec5).float()/(index*opt.test_batch_size)
    
    avg_test_loss = avg_test_loss/(index)
    
    print("Testing Prec1:{}%, Prec5:{}%".format(avg_test_prec1, avg_test_prec5))
    
    return avg_test_prec1, avg_test_prec5, avg_test_loss
      
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=True):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id, set_affinity = True)
        self.input = ops.FileReader(file_root=data_dir, num_shards = device_id+1, shard_id=device_id, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        device_memory_padding=211025920 if decoder_device == 'mixed' else 0
        host_memory_padding=140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device = decoder_device, output_type = types.RGB,
                                                device_memory_padding = device_memory_padding,
                                                host_memory_padding = host_memory_padding,
                                                random_aspect_ratio = [0.8, 1.25],
                                                random_area = [0.08, 1.0],
                                                num_attempts = 100)
        
        self.res = ops.Resize(device = dali_device, resize_x=crop, resize_y=crop, interp_type = types.INTERP_TRIANGULAR)
        
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        
        return [output, self.labels]
        
        
class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed = 12 + device_id, set_affinity = True)
        
        self.input = ops.FileReader(file_root=data_dir, num_shards = device_id+1, shard_id=device_id,  random_shuffle=False)
        self.decode = ops.ImageDecoder(device = 'mixed', output_type = types.RGB)
        #self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device = 'gpu', resize_shorter = size, interp_type = types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        #self.iteration += 1
        #if self.iteration % 200 == 0:
        #  del images, self.jpegs
        return [output, self.labels]
        
def get_data(opt):
  if not opt.dataset == 'ImageNet':
    transform_train = input_transforms['{}'.format(opt.dataset)]['train']
    transform_test = input_transforms['{}'.format(opt.dataset)]['eval']
    
    if opt.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='data/MNIST/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='data/MNIST/', train=False, download=True, transform=transform_test)
        
    elif opt.dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/CIFAR10/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='data/CIFAR10/', train=False, download=True, transform=transform_test)
    
    elif opt.dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/CIFAR100/', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='data/CIFAR100/', train=False, download=True, transform=transform_test)
      
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_batch_size, shuffle = True, num_workers=opt.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    return trainloader, testloader     
  
  else:
    traindir = os.path.join(opt.imagenet_directory, 'train')
    valdir = os.path.join(opt.imagenet_directory, 'val')

    crop_size = 224
    val_size = 256
      
    pipe = HybridTrainPipe(batch_size=opt.train_batch_size, num_threads=opt.num_workers, device_id=int(opt.gpu), data_dir=traindir, crop=crop_size, dali_cpu=opt.dali_cpu)
    pipe.build()
    trainloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / 1), auto_reset = True)

    pipe = HybridValPipe(batch_size=opt.test_batch_size, num_threads=opt.num_workers, device_id=int(opt.gpu), data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    testloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / 1), auto_reset = True)
  
    return trainloader, testloader     

def get_model(opt, num_classes, mode = "student", model = "boolnet18"):
    if mode in {"student", "pretrain"}:

      from models.boolnet import boolnet18, boolnet34
      
      if model == 'boolnet18':
        net = boolnet18(num_classes = num_classes, max_slices = opt.max_slices, binary_downsample = opt.binary_downsample)
    
      elif model == 'boolnet34':
        net = boolnet34(num_classes = num_classes, max_slices = opt.max_slices, binary_downsample = opt.binary_downsample)
      
      else:
        raise ValueError
      
      
    else:
      from models.resnet_org import resnet18, resnet34, resnet50, resnet101, resnet152

      pretrained = False if opt.dataset != "ImageNet" else True

      print("=======> Loading teacher {} with pretrained={}".format(model, pretrained))

      if model == 'resnet18':
        net = resnet18(pretrained=pretrained)

      elif model == 'resnet34':
        net = resnet34(pretrained=pretrained)

      elif model == 'resnet50':
        net = resnet50(pretrained=pretrained)

      elif model == 'resnet101':
        net = resnet101(pretrained=pretrained)

      elif model == 'resnet152':
        net = resnet152(pretrained=pretrained)

      else:
        raise ValueError

      if opt.dataset != "ImageNet":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
      else:
        pass
          
    return net
        
def get_optimizer(opt, parameters, mode = "train"):
    
    if opt.optimizer == 'Adam':
      optimizer = optim.Adam(
                        parameters, 
                        lr = opt.lr,
                        weight_decay = opt.weight_decay,
                        betas = (0.9, 0.999),
                        amsgrad = False, 
                        )
    
    elif opt.optimizer == "RAdam":
        optimizer = RAdam(
                        parameters,
                        lr = opt.lr,
                        weight_decay = opt.weight_decay,
                        degenerated_to_sgd = False,
                        )

    elif opt.optimizer == 'SGD':
      optimizer = optim.SGD(
                        parameters, 
                        optim_params['SGD'][opt.dataset]['Init_lr'], 
                        momentum=optim_params['SGD'][opt.dataset]['Weight_momentum'], 
                        weight_decay=optim_params['SGD'][opt.dataset]['Weight_decay'],
                        nesterov  = False
                        )
                       
    else:
      raise ValueError('Only support Adam or SGD !!')
    
    if opt.lr_scheduler == "multisep":
      lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=optim_params[opt.optimizer][opt.dataset]['MultiStepLR']['step'], gamma=optim_params[opt.optimizer][opt.dataset]['MultiStepLR']['ratio'])
    
    elif opt.lr_scheduler == "cosine": 
      lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch, eta_min = 1e-8)
    
    elif opt.lr_scheduler == "exponential":
      lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96)
    
    elif opt.lr_scheduler == "plateau":
      lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", threshold = 5e-2, factor = 0.5, patience = 20, min_lr = 1e-10)
          
    return optimizer, lr_scheduler   
    
def set_device(opt, net, optimizer):  
    if opt.cuda:
      if opt.parallel:
        gpus = list(range(opt.parallel))

        if net is not None:
          if opt.acceleration:
            print("FP 16 Trianing")
            amp.register_float_function(torch, 'sigmoid')
            net, optimizer = amp.initialize(net.cuda(), optimizer, opt_level="O1")
            
          net = nn.DataParallel(net, gpus).cuda()
          
        else:
          raise ValueError('The network is missing')
      else:
        net = net.cuda()
        if opt.acceleration:
            print("FP 16 Trianing")
            amp.register_float_function(torch, 'sigmoid')
            net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    
    else:
      net = net.cpu()
    
    return net

def set_temperature_cifar(net, opt, epochs, freq = 5):
    Temperature = torch.ones(1)
    for i, module in enumerate(net.module.modules() if opt.parallel else net.modules()):
        if isinstance(module, (BConv, MultiBConv, GhostBNSign, GhostSign)):
          if epochs != 0:
            module.update_temperature()
          
          Temperature = torch.Tensor([module.temperature])
        
    return Temperature

def set_temperature_imagenet(net, opt, epochs, freq = 5):
    Temperature = torch.ones(1)
    for i, module in enumerate(net.module.modules() if opt.parallel else net.modules()):
        if isinstance(module, (BConv, MultiBConv, GhostBNSign, GhostSign)):
          if epochs != 0 and epochs % freq == 0:
            module.update_temperature()
          
          Temperature = torch.Tensor([module.temperature])
        
    return Temperature

      
def load_state_dict(net, train_optimizer, train_lr_scheduler, pretrained_dir):
    if os.path.exists(pretrained_dir):
      weights = torch.load(pretrained_dir)
      
      try:
        net.load_state_dict(weights["network"])
        if train_optimizer is not None:
          train_optimizer.load_state_dict(weights["train_optimizer"])
        else:
          pass
        
        if train_lr_scheduler is not None:
          train_lr_scheduler.load_state_dict(weights["train_lr_scheduler"])
        else:
          pass
            
        print("=======> loading model '{}'".format(pretrained_dir))
        
      except:
        raise ValueError("the pretrained state_dict does not match current architecture !")
      
      return weights
    else:
        raise ValueError("the pretrained model pasth does not exists !")

def save_state_dict(net, train_optimizer, train_lr_scheduler, epoch, opt, best_acc, running_epochs, running_best_acc, running_train_prec1, running_train_prec5, running_train_loss, running_test_prec1, running_test_prec5, running_test_loss):
    if not os.path.exists("checkpoints"):
      os.mkdir("checkpoints")
    
    if not os.path.exists("checkpoints/{}/".format(opt.dataset)):
      os.mkdir("checkpoints/{}/".format(opt.dataset))
    
    if not os.path.exists("checkpoints/{}/{}/".format(opt.dataset, opt.model)):
      os.mkdir("checkpoints/{}/{}/".format(opt.dataset, opt.model))
    
    net = net.cpu()
        
    checkpoints = {
                "epoch":epoch,
                "network":net.state_dict(), 
                "train_optimizer":train_optimizer.state_dict(), 
                "train_lr_scheduler":train_lr_scheduler.state_dict(),
                "epoch":epoch,
                "best_acc":best_acc,
                "running_epochs": running_epochs,
                "running_best_acc":running_best_acc,
                "running_train_prec1":running_train_prec1,
                "running_train_prec5":running_train_prec5,
                "running_train_loss":running_train_loss,
                "running_test_prec1":running_test_prec1,
                "running_test_prec5":running_test_prec5,
                "running_test_loss":running_test_loss
                }
    
    name = "checkpoints/{}/{}/{}_{}_lablesmooth_{}_mixup_{}_slices_{}_best.pth".format(opt.dataset, opt.model, opt.model, opt.mode, opt.labelsmoothing, opt.mixup, opt.max_slices)
    
    torch.save(checkpoints, name)
    
    net.cuda()
    
    
    print("======> Save checkpoints in " + name)
    
    return best_acc

def accuracy(output, target, topk = (1,)):
    maxk = 5
    
    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum().cpu()
      res.append(correct_k)
    
    return res
    
    
if __name__ == "__main__":
  main()        


