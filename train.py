import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2

from utils import load_Anime
from model import discriminator, generator

# change requires_grad of certain net
def set_requires_grad(net, switch):
    for param in net.parameters():
        param.requires_grad = switch
    return

# change to cuda Variable
def to_var(x, requires_grad=True):
    x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def to_var2(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.FloatTensor
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))  # alpha:(batch_size,1,1,1)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)  # real和fake之间选一点
    d_interpolates = D(interpolates)  # (batch, 1)  # need to be revised!!
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def train(G, D,
          optimizer_G, optimizer_D, criterion,
          epoches, z_dim,
          batch_size=32, LAMBDA=0.1, use_GP = False):

    np.random.seed(9487)
    random.seed(9487)
    #torch.set_random_seed(9487)

    # tip 3: Use spherical z(noise_vector)
    #   Dont sample from a Uniform distribution.
    #   Sample from a gaussian distribution.
    #   When doing interpolations, do the interpolation via a great circle,
    #   rather than a straight line from point A to point B.
    sample_z = np.random.normal(0, np.exp(-1/np.pi), size=(batch_size, z_dim))

    test_labels = np.zeros((batch_size, 23)) # y_dim=23
    # what we want
    # blue hair, green eye.
    for i in range(5):
        test_labels[5 + i][8] = 1
        test_labels[5 + i][19] = 1
    # blue hair, red eye.
    for i in range(5):
        test_labels[10 + i][8] = 1
        test_labels[10 + i][21] = 1
    # green hair, blue eye.
    for i in range(5):
        test_labels[15 + i][4] = 1
        test_labels[15 + i][22] = 1
    # green hair, red eye.
    for i in range(5):
        test_labels[20 + i][4] = 1
        test_labels[20 + i][21] = 1

    start_time = time.time()
    #g_loss = 0.0

    data_X, data_Y = load_Anime('../extra_data/images/')  # takes about 2 min

    # used for debugging!!
    #data_X = np.ones((150,64,64,3))
    #data_Y = np.ones((150,23))

    length = len(data_X)
 
    num_batches = (int)(length/batch_size)
    for epoch in range(epoches):
        counter = 0
        for idx in range(num_batches-1):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            #print('i am here')
            batch_images = np.asarray(data_X[idx * batch_size:(idx+1) * batch_size]).astype(np.float32).transpose(0,3,1,2)
            batch_labels = np.asarray(data_Y[idx * batch_size:(idx+1) * batch_size]).astype(np.float32)
            batch_images_wrong = np.asarray(data_X[random.sample(range(len(data_X)), len(batch_images))]).astype(np.float32).transpose(0,3,1,2)
            batch_labels_wrong = np.asarray(data_Y[random.sample(range(len(data_Y)), len(batch_images))]).astype(np.float32)

            batch_images = to_var(batch_images, False)
            batch_labels = to_var(batch_labels, False)
            batch_images_wrong = to_var(batch_images_wrong, False)
            batch_labels_wrong = to_var(batch_labels_wrong, False)
            # Tip 3. Use a spherical Z
            #   Dont sample from a Uniform distribution.
            #   Sample from a gaussian distribution.
            #   When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B.
            #   Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details.
            batch_z = np.random.normal(0, np.exp(-1/np.pi), size=[batch_size, z_dim]).astype(np.float32)
            batch_z = to_var(batch_z)
            # update G network
            # First set all discriminator to requires_grad=False
            set_requires_grad(D, False)
            # Then run G
            fake_imgs = G(batch_z, batch_labels)  # (b, 3,64,64)
            # get loss function for G
            g_score = D(fake_imgs, batch_labels)
            
            g_loss = criterion(g_score, to_var2(torch.FloatTensor(round(batch_size), 1).fill_(1.0), requires_grad=False))  # changed!!
            #print('i am here')
            if (epoch + 1) % 4 == 0 or epoch > 30:      # 每2次就做一次bp，更新generator(少训练generator)
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

            set_requires_grad(D, True)
            # compute loss function for D
            D_fake = D(fake_imgs, batch_labels)
            D_real = D(batch_images, batch_labels)
            D_wrong_img = D(batch_images_wrong, batch_labels)
            D_wrong_label = D(batch_images, batch_labels_wrong)

            d_loss_real = criterion(D_real, to_var2(torch.FloatTensor(round(batch_size), 1).fill_(1.0), requires_grad=False))

            d_loss_fake_1 = criterion(D_fake, to_var2(torch.FloatTensor(round(batch_size), 1).fill_(0.0), requires_grad=False))
            d_loss_fake_2 = criterion(D_wrong_img, to_var2(torch.FloatTensor(round(batch_size), 1).fill_(0.0), requires_grad=False))
            d_loss_fake_3 = criterion(D_wrong_label, to_var2(torch.FloatTensor(round(batch_size), 1).fill_(0.0), requires_grad=False))
            d_loss = d_loss_real + (d_loss_fake_1 + d_loss_fake_2 + d_loss_fake_3)/3
            #print('i am here')
            if use_GP:
                gradient_penalty = compute_gradient_penalty(D, batch_images, fake_imgs)
                d_loss += LAMBDA * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            counter += 1
            if counter % 2 == 0:
                print("Epoch: [%4d/%4d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (epoch, epoches, idx, num_batches, time.time() - start_time, d_loss, g_loss), end='\r')
                print()
                
        if epoch%10 == 0:
            im_fake_output = np.clip(fake_imgs.cpu().data.numpy(), -1, 1)
            im_fake_output = np.round((im_fake_output+1) * 127.5) #
            im_fake_output = np.transpose(im_fake_output, (0,2,3,1))  # (b,3,64,64) -> (b,64, 64, 3)
            for i in range(batch_size):
                cv2.imwrite("./imgs/picture_epoch{}_{}.jpg".format(epoch + 1, i), im_fake_output[i])
            print('Save imgs successfully!!')
            torch.save(G.state_dict(), './models/generator_epoch{}.pkl'.format(epoch + 1))
            print('Save model successfully!!')


def main():
    z_dim = 128
    D = discriminator()
    G = generator(z_dim=128)
    if torch.cuda.is_available():
        print('Using CUDA')
        D = D.cuda()                       # model.cuda()
        G = G.cuda()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)  # use Adam
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002) # use Adam
    criterion = nn.BCELoss()
    train(G, D,
          optimizer_G, optimizer_D, criterion,
          200, z_dim,
          batch_size=32, LAMBDA=0.25, use_GP = False) 

if __name__ == '__main__':
    main()













