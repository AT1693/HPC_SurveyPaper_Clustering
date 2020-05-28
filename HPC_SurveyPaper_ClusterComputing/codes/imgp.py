import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
a=['wheat_grain21.jpeg','wheat_grain27.jpeg','wheat_grain32.jpeg','wheat_grain37.jpeg']
data_path='/home/ambuje/Desktop/'
import mpi4py.MPI
rank = mpi4py.MPI.COMM_WORLD.Get_rank()
size = mpi4py.MPI.COMM_WORLD.Get_size()

task_list = range(4)
aa=time.time()

def f(task,i):
    q=data_path+a[i]
    img = cv2.imread(q,0)
#     def auto_canny(image, sigma=0.33):
#         # compute the median of the single channel pixel intensities
#         v = np.median(image)
#         # apply automatic Canny edge detection using the computed median
#         lower = int(max(0, (1.0 - sigma) * v))
#         upper = int(min(255, (1.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         # return the edged image
#         return edged
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    #wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    #auto = auto_canny(blurred)
    # show the images
    # cv2.imshow("Original", img)
    # cv2.imshow("Edges", np.hstack([wide, tight, auto]))
    # cv2.waitKey(0)
    (cnts, _) = cv2.findContours(tight, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
    print("I found %i wheat grains" % len(cnts))
for i,task in enumerate(task_list):
  #This is how we split up the jobs.
  #The % sign is a modulus, and the "continue" means
  #"skip the rest of this bit and go to the next time
  #through the loop"
  # If we had e.g. 4 processors, this would mean
  # that proc zero did tasks 0, 4, 8, 12, 16, ...
  # and proc one did tasks 1, 5, 9, 13, 17, ...
  # and do on.
  if i%size!=rank: continue
  #print ("Task number %d (%d) being done by processor %d of %d" % (i, task, rank, size))
  f(task,i)
b=time.time()
print(b-aa)
