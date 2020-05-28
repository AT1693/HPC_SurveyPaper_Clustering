import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
import mpi4py.MPI
rank = mpi4py.MPI.COMM_WORLD.Get_rank()
size = mpi4py.MPI.COMM_WORLD.Get_size()
l=['wheat_grain27.jpeg','wheat_grain32.jpeg','wheat_grain21.jpeg','wheat_grain37.jpeg']
task_list = range(4)
aa=time.time()
def f(task,i):
  img = cv2.imread(l[i],1)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  # noise removal
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

  # sure background area
  sure_bg = cv2.dilate(opening,kernel,iterations=3)

  # Finding sure foreground area
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)


  # Finding unknown region
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)

  # Marker labelling
  ret, markers = cv2.connectedComponents(sure_fg)

  # Add one to all labels so that sure background is not 0, but 1
  markers = markers+1

  # Now, mark the region of unknown with zero
  markers[unknown==255] = 0

  markers = cv2.watershed(img,markers)
  img[markers == -1] = [255,0,0]


  #thresholding a color image, here keeping only the blue in the image
  th=cv2.inRange(img,(255,0,0),(255,0,0)).astype(np.uint8)


  #inverting the image so components become 255 seperated by 0 borders.
  th=cv2.bitwise_not(th)

  #calling connectedComponentswithStats to get the size of each component
  nb_comp,output,sizes,centroids=cv2.connectedComponentsWithStats(th,connectivity=4)

  #taking away the background
  nb_comp-=1; sizes=sizes[0:,-1]; centroids=centroids[1:,:]

  bins = list(range(np.amax(sizes)))

  #plot distribution of your cell sizes.

  numbers = sorted(sizes)
  fin=len(numbers)-2


  # plt.hist(sizes,numbers)
  # s="result_"+str(i)+".jpeg"
  # cv2.imwrite(s,img)

  print(fin)
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







