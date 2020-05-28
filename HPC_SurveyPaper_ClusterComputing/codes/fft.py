import pandas as pd
import cv2
import numpy as np
import time
import mpi4py.MPI
rank = mpi4py.MPI.COMM_WORLD.Get_rank()
size = mpi4py.MPI.COMM_WORLD.Get_size()

q=pd.read_csv('eeg_raw_values.csv').to_numpy()
#print(q)

task_list = range(len(q))
z=time.time()
def fft_v(task,i):
	x=q[i]
	x = np.asarray(x, dtype=float)
	N = x.shape[0]
	if np.log2(N) % 1 > 0:
		raise ValueError("must be a power of 2")

	N_min = min(N, 2)

	n = np.arange(N_min)
	k = n[:, None]
	M = np.exp(-2j * np.pi * n * k / N_min)
	X = np.dot(M, x.reshape((N_min, -1)))
	while X.shape[0] < N:
		X_even = X[:, :int(X.shape[1] / 2)]
		X_odd = X[:, int(X.shape[1] / 2):]
		terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
			        / X.shape[0])[:, None]
		X = np.vstack([X_even + terms * X_odd,
			       X_even - terms * X_odd])
	return X.ravel()
def fft():
	x=q[i]
	x = np.asarray(x, dtype=float)
	N = x.shape[0]
	if np.log2(N) % 1 > 0:
		raise ValueError("must be a power of 2")

	N_min = min(N, 2)

	n = np.arange(N_min)
	k = n[:, None]
	M = np.exp(-2j * np.pi * n * k / N_min)
	X = np.dot(M, x.reshape((N_min, -1)))
	while X.shape[0] < N:
		X_even = X[:, :int(X.shape[1] / 2)]
		X_odd = X[:, int(X.shape[1] / 2):]
		terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
			        / X.shape[0])[:, None]
		X = np.vstack([X_even + terms * X_odd,
			       X_even - terms * X_odd])
	return X.ravel()

for i,task in enumerate(task_list):
  #print(i,task)
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
  fft_v(task,i)
b=time.time()
print("PARALLEL",b-z)

