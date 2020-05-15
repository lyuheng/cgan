import torch
import numpy as np
import cv2

a = torch.Tensor([[1,2,2],[2,2,2]])
#print(a.repeat(2,2).shape)
#print(a.reshape((-1,2)))
#print(len(a))

im_fake_output = np.ones((32, 64,64,3))*128
cv2.imwrite("./imgs/picture.jpg" , im_fake_output[1])