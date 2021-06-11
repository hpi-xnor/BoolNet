import torch.nn as nn
from torchvision.transforms import transforms
from torchtoolbox.transform import Cutout

optim_params = {
  'Adam': {'CIFAR10': {'Init_lr':1e-3,
                       'Betas': (0.9, 0.999),
                       'Weight_decay': 0,
                       'MultiStepLR': {'step': [30, 60, 90, 120, 150, 180], 'ratio': 0.2}
                      },
                              
           'CIFAR100': {'Init_lr':1e-3,
                        'Betas': (0.9, 0.999),
                        'Weight_decay': 0,
                        'MultiStepLR': {'step': [30, 60, 90, 120, 150, 180], 'ratio': 0.2}
                       },
           
           'ImageNet': {'Init_lr':1e-3,
                        'Betas': (0.9, 0.999),
                        'Weight_decay': 0,
                        'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.5}  
                       },
          },
          
  'SGD': {'CIFAR10': {'Init_lr': 0.1,
                      'Weight_momentum':0.9,
                      'Weight_decay': 0,
                      'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.1},
                     },
                               
          'CIFAR100': {'Init_lr': 0.1,
                       'Weight_momentum':0.9,
                       'Weight_decay': 0,
                       'MultiStepLR': {'step': [60, 120, 180, 240, 300, 360, 420, 480], 'ratio': 0.1},
                      },
                  
          },            
  }
        
input_transforms = {
 'CIFAR10': {
          'train': transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout()
              ]),
            
          'eval': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
              ])},
            
 'CIFAR100':{
          'train': transforms.Compose([
                transforms.RandomCrop(32, padding = 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ]),
            
          'eval': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)) 
              ])},
              
 'ImageNet':{
          'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)) 
              ]),
            
          'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.255)) 
              ])},
                  
 }
