import skimage
import skimage.io
import numpy as np
import zipfile, io
import pygame
import os


# returns image of shape [400, 400, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    print(img[0]*255)
    assert(img.shape==(400,400,3))
    return img

def load_images_from_zip(path):
    archive = zipfile.ZipFile(path, 'r')
    name_list=archive.namelist()
    images=np.zeros((len(name_list)-1,400,400,3))
    idx=0
    for name in name_list:
        if (name[len(name)-1]=='/'):
            continue
        img_data = archive.read(name)
        bytes_io = io.BytesIO(img_data)
        img = pygame.image.load(bytes_io)
        img = pygame.surfarray.array3d(img)
        images[idx]=img
        idx+=1
    archive.close()
    return images
def load_image_rwanda_zip(name):
    archive = zipfile.ZipFile("rwanda.zip", 'r')
    img_data = archive.read("rwanda/"+name)
    bytes_io = io.BytesIO(img_data)
    img = pygame.image.load(bytes_io)
    img = pygame.surfarray.array3d(img)
    archive.close()
    return img
# posibilité d'utiliser images=np.concatenate((images, img), 0)
def load_images(path_dir):
    path_list=os.listdir(path_dir)
    images=np.zeros(len(path_list),400,400,3)
    idx=0
    for path in path_list:
        img=load_image(path_dir+path)
        images[idx]=img
        idx+=1
    return images

# returns the top1 prob
def print_prob(prob):
    # print prob
    pred = np.argsort(prob)[::-1]
    #print the top1 prob
    print("Prediction: y=%d avec une proba %d" % (pred[0],prob[pred[0]]))
    return prob[pred[0]]
