import torch
import numpy as np
import matplotlib.pyplot as plt
from model import generator


def save_imgs(gen_imgs):
    r, c = 5, 5
    # gen_imgs should be shape (25, 64, 64, 3)
    # gen_imgs = generator(noise)
    fig, axs = plt.subplots(r, c)    # 5*5 subplot
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])  # 25 pics
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output.png")
    plt.close()



def main():
    test_labels = np.zeros((25, 23)) # y_dim=23  # labels
    # what we want to generate
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

    sample_z = np.random.normal(0, np.exp(-1/np.pi), size=(25, 128)) # batch_size=20 # noise_dim=128
    sample_z = torch.FloatTensor(sample_z).cuda()
    test_labels = torch.FloatTensor(test_labels).cuda()
    G = generator(z_dim=128).cuda()
    print('loading params into generator!')
    G.load_state_dict(torch.load('./models/generator_epoch191.pkl'))
    print('loading params successfully!')
    print('start generating images!')
    imgs = G(sample_z, test_labels)
    imgs = imgs.cpu().data.numpy()
    #imgs = np.round((imgs+1) * 127.5) 
    imgs = np.transpose(imgs, (0,2,3,1))  # (b,3,64,64) -> (b,64, 64, 3)
    save_imgs(imgs)
    print('finish!')


if __name__ == '__main__':
    main()
    
    
