# This code is modified from
# https://github.com/mibastro/VPE
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.misc as m
import random
import csv
class hifLoader(Dataset):

  def __init__(self, root, exp, split='train', is_transform=False, img_size=None, augmentations=None, prototype_sampling_rate=0.005):
    super().__init__()

    if split == 'train':
        self.proto_rate = prototype_sampling_rate
    else:
        self.proto_rate = 0.0
        
    self.inputs = []
    self.targets = []
    self.class_names = []
    self.split = split
    self.img_size = img_size
    self.is_transform = is_transform
    self.augmentations = augmentations
    self.mean = np.array([125.00, 125.00, 125.00]) # average intensity

    self.root = root + 'hifsignals/'
    exp  = exp + '/exp_hifsignals/'
    #_impaths.txt labels path data to individual files
    self.dataPath = root + exp + self.split + '_impaths.txt'
    #_imclasses.txt labels the classes for each image in numerical order 
    self.labelPath = root + exp + self.split + '_imclasses.txt'

    self.tr_class = torch.LongTensor([1,2,3,4,5,7,8,9,10,11,12,13,14,15,17,18,25,26,31,33,35,38])
    self.te_class = torch.LongTensor([0,6,16,19,20,21,22,23,24,27,28,29,30,32,34,36,37,39,40,41,42])

    f_data = open(self.dataPath,'r')
    f_label = open(self.labelPath,'r')
    data_lines = f_data.readlines()
    label_lines = f_label.readlines()

    for i in range(len(data_lines)):
      self.inputs.append(root+data_lines[i][0:-1])
      self.targets.append(int(label_lines[i].split()[0])) # label: [road class, wet/dry, video index]

    #_classnames.txt contains the plaintext names of the classes
    classnamesPath = root + exp + '/classnames.txt'
    f_classnames = open(classnamesPath, 'r')
    data_lines = f_classnames.readlines()
    for i in range(len(data_lines)):
        self.class_names.append(data_lines[i][0:-1])
    # n_classes is the number of labeled classes in the particular dataset - gtsrb has 43, should be updated accordingly
    self.n_classes = 43
    assert(self.n_classes == len(self.class_names))

    print('HIF %d classes'%(len(self.class_names)))
    print('Load HIF %s: %d samples'%(split, len(self.targets)))


  def __len__(self):
    return len(self.inputs)


  def __getitem__(self, index):
    signal_path = self.inputs[index]
    gt = self.targets[index]
    gt = torch.ones(1).type(torch.LongTensor)*gt

    # Load images and templates. perform augmentations
    # these will need to be changed to CSV readers to extract voltage/current values
    #img = m.imread(signal_path)
    with open(self.dataPath, 'r') as f:
        data_lines = f.readlines()

    img = np.array(img, dtype=np.uint8)
    template = m.imread(self.root + 'template_ordered/%02d.jpg'%(gt+1))
    template = np.array(template, dtype=np.uint8)

    if random.random() < self.proto_rate:
        img = np.copy(template)

    if self.augmentations is not None:
        img, template = self.augmentations(img, template)

    if self.is_transform:
        img = self.transform(img)
        template = self.transform(template)

    return img, gt, template

    # transform function for preprocessing images -- what should we do for signal data?
  def transform(self, img):
    img = img.astype(np.float64)
    img -= self.mean
    if self.img_size is not None:
      img = m.imresize(img, (self.img_size[0], self.img_size[1]))
    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    
    return img

#these are the "prototypes" we're using for training
  def load_template(self, target, augmentations=None):

    # if augmentation is not specified, use self.augmentations. Unless use input augmentation option.
    # augmentations are defined in augmentations.py, only "transform" we could do added noise for signal data like in the HIF detection paper
    if augmentations is None:
        augmentations = self.augmentations
    img_paths = []

    #gets prototype image paths - will need to be updated to get paths to prototype signal files
    for id in target:
        img_paths.append(self.root + '/template_ordered/%02d.jpg'%(id+1))

    target_img = []
    for img_path in img_paths:
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if augmentations is not None:
            img, _ = augmentations(img, img)
        if self.transform:
            img = self.transform(img)

        target_img.append(img)

    return torch.stack(target_img, dim=0)