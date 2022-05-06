""" Use torchIO to augment the dataset and save to a list
Typical usage example:
list = torchio_data.tioData(dir_img, dir_mask)
"""
import torch
import torchio as tio
from PIL import Image
import torch
from torchvision.transforms import functional as tvF
import cv2
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
import os
import torchvision

def tensorToCV(img):
    img =img.T.swapaxes(0, 1)
    srcimg = img.numpy()/1.5*255
    imgr = np.where(srcimg[:,:,0]>0,srcimg[:,:,0],-srcimg[:,:,0])
    imgg = np.where(srcimg[:,:,1]>0,srcimg[:,:,1],-srcimg[:,:,1])
    imgb = np.where(srcimg[:,:,2]>0,srcimg[:,:,2],-srcimg[:,:,2])
    srcimg[:,:,0] = imgb
    srcimg[:,:,1] = imgg
    srcimg[:,:,2] = imgr
    srcimg = srcimg.astype('uint8')
    return srcimg


"""Generate torchIO Subject data list
    Read specified original image dir and mask dir and load to a torchIO Subject list
    Args:
        train_src: trainning images dir.
        train_label: trainning label/mask dir.
    Returns:
        A list of torchIO Subject with the user specified trainning images and mask
    Raises:
        IOError: An error occurred accessing the Eempty dir.
    """
def tioData(train_src,train_label):
    subject_list = []
    if len(train_src) == len(train_label):  # if size of images and masks are same
        # sort as the file names
        train_src.sort()
        train_label.sort()
        # read image and mask one by one
        for index in range(len(train_src)):
            # read image and convert to grat
            img = Image.open(train_src[index])
            lbl = Image.open(train_label[index]).convert('L')
            # convert to pytorch.tensor
            image =tvF.to_tensor(img)
            mask = tvF.to_tensor(lbl)
            # convert to 4D tensor fore tio
            tioimage = torch.unsqueeze(image,3)
            tiomask = torch.unsqueeze(mask,3)
            # convert to TIO subject
            subject_vessel = tio.Subject(
                t1=tio.ScalarImage(tensor=tioimage),
                label=tio.LabelMap(tensor=tiomask),
                diagnosis='negative',
            )
            subject_list.append(subject_vessel)
    else:
        return None
    print("Original Data Size: ",len(subject_list))
    transforms = tio.Compose([
            tio.RandomAffine(scales=(0.9, 1.1),degrees=20,isotropic=True,image_interpolation='nearest',),
            tio.RandomElasticDeformation(num_control_points=10,max_displacement=30,locked_borders=2),
            tio.RandomBiasField(coefficients=(0,0.2),order=2),
            tio.RandomNoise(mean=0,std=(0, 0.001)),
            tio.RandomFlip(axes=('LR')),
            tio.RandomGamma(log_gamma=(-0.3, 0.3))
            ])
    subjects_dataset = tio.SubjectsDataset(subject_list, transform=transforms) 
    return subjects_dataset

if __name__ == '__main__':
    augmenttimes = 1
    visualization = True
    filename_index = 0

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    train_src = glob.glob('./data/imgs/*.png')
    train_label = glob.glob('./data/masks/*.png')
    
    print(train_src)
    for n in range(augmenttimes):
        subjects_dataset = tioData(train_src,train_label)
        training_loader = DataLoader(subjects_dataset, batch_size=1, num_workers=0)
        batch_num = 1
        for batch in training_loader:
            images = batch['t1'][tio.DATA]
            labels = batch['label'][tio.DATA]
            for index in range(len(images)):
                tile = 'Round',n,"_index_",index

                imgsrc = images[index].squeeze()
                imglabel = labels[index].squeeze()
                if visualization:
                    plt.subplot(211)  
                    plt.imshow(imgsrc.T.swapaxes(0, 1))
                    plt.title(tile) 
                    plt.subplot(212)  
                    plt.imshow(imglabel)
                    plt.show() 

                srcimg = tensorToCV(imgsrc)

                img =imglabel.T.swapaxes(0, 1)
                img = img.numpy()*255
                labelimg = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
                labelimg[:,:,0] = np.where(img>250,255,0)
                labelimg[:,:,1] = np.where(img>0,255,0)

                cv2.imwrite("./newdata/imgs/{:0>5d}.png".format(filename_index),srcimg)
                cv2.imwrite("./newdata/masks/{:0>5d}.png".format(filename_index),labelimg)
                filename_index = filename_index+1
                