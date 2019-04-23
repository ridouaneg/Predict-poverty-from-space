import skimage
import skimage.io
import skimage.transform
import numpy as np
import os


# returns image of shape [400, 400, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    assert(img.shape==(400,400,3))
    return img

def load_images(path_dir):
    images=np.array([])
    idx=0
    for path in os.listdir(path_dir):
        img=load_image(path_dir+path)
        images=batch = np.concatenate((images, img), 0)
        idx+=1
    return images

# returns the top1 prob
def print_prob(prob):
    # print prob
    pred = np.argsort(prob)[::-1]
    #print the top1 prob
    print("Prediction: y=%d avec une proba %d" % (prob,prob[pred[0]]))
    return prob[pred[0]]
