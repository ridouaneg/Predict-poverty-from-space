#Useless script
import utils
import numpy as np
import time

debut=time.time()
A=utils.load_images("data/")
fin=time.time()
print(fin-debut)
