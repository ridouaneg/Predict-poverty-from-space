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
    img = skimage.io.imread(path)[:,:,3]
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
    images=np.zeros((len(path_list),400,400,3))
    idx=0
    for path in path_list:
        img=load_image(path_dir+path)
        images[idx]=img
        idx+=1
    return images

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def prediction(prob):
    pred = np.argmax(prob)
    print("Prediction: y={} avec une proba {}".format(pred,prob[pred]))
    return pred
