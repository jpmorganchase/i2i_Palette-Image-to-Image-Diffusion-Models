# ----------------------------------------------------------------------------
# Adapted from https://github.com/mxl1990/tsne-pytorch/blob/master/tsne_torch.py
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")
parser.add_argument("--ckpt_folder", type=str, default=None, help="if use cuda accelarate")
parser.add_argument('--mask_ratio', default=0.5, type=float, help='masking ratio w.r.t. one dimension')

opt = parser.parse_args()
print("get choice from args", opt)

if opt.cuda:
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    # this part may be some difference for complex eigenvalue
    # but complex eignevalue is meanless here, so they are replaced by their real part
    i = 0
    while i < d:
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 2
        else:
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    # X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    class_list = open('datasets/place365/flist/class.txt').read().splitlines()
    retain_class = class_list[:50]
    forget_class = class_list[50:100]
    img_folder = opt.ckpt_folder
    retain_imgs=[]
    for retainname in retain_class:
        tmp = glob.glob(os.path.join(img_folder, f'results/test/0/{opt.mask_ratio:.2f}/Out'+retainname.replace('/', '_')+'_Places365*.jpg'))
        retain_imgs += tmp
    retain_imgs = sorted(retain_imgs)
    
    forget_imgs=[]
    for forgetname in forget_class:
        tmp = glob.glob(os.path.join(img_folder, f'results/test/0/{opt.mask_ratio:.2f}/Out'+forgetname.replace('/', '_')+'_Places365*.jpg'))
        forget_imgs += tmp
    forget_imgs = sorted(forget_imgs)

    our_forget = os.path.join(opt.ckpt_folder, 'forget_clip_norm.txt')
    our_retain = os.path.join(opt.ckpt_folder, 'retain_clip_norm.txt')
    forget_norm = np.loadtxt(our_forget)
    retain_norm = np.loadtxt(our_retain)
    base_forget = np.zeros_like(forget_norm)
    base_retain = np.zeros_like(retain_norm)

    base_img_folder = './ckpt/original'
    for imgid, imgname in enumerate(forget_imgs):
        base_name = os.path.basename(imgname)
        base_forget[imgid] = np.loadtxt(os.path.join(base_img_folder, base_name.replace('.jpg', '_clip_norm.txt').replace('Out_', '')))
    for imgid, imgname in enumerate(retain_imgs):
        base_name = os.path.basename(imgname)
        base_retain[imgid] = np.loadtxt(os.path.join(base_img_folder, base_name.replace('.jpg', '_clip_norm.txt').replace('Out_', '')))

    unlearn_forget_norm = forget_norm[::100, ...]
    unlearn_retain_norm = retain_norm[::100, ...]
    original_forget_norm = base_forget[::100, ...]
    original_retain_norm = base_retain[::100, ...]


    unlearn_forget_label, unlearn_retain_label = np.zeros(unlearn_forget_norm.shape[0])+0.0, np.zeros(unlearn_retain_norm.shape[0])+1.0
    original_forget_label, original_retain_label = np.zeros(original_forget_norm.shape[0])+2.0, np.zeros(original_retain_norm.shape[0])+3.0

    X = np.concatenate((unlearn_forget_norm, unlearn_retain_norm, original_forget_norm, original_retain_norm), axis=0)

    del unlearn_forget_norm
    del unlearn_retain_norm
    del original_forget_norm
    del original_retain_norm
    X = torch.Tensor(X)
    labels = np.concatenate((unlearn_forget_label, unlearn_retain_label, original_forget_label, original_retain_label), axis=0).tolist()

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0])==len(X[:,1]))
    assert(len(X)==len(labels))

    with torch.no_grad():
        Y = tsne(X, 2, 50, 20.0)

    if opt.cuda:
        Y = Y.cpu().numpy()

    plt.rcParams['font.size']=20
    plt.rcParams['figure.figsize']=(8,6)

    fig, ax = plt.subplots()
    mi=['s','s','x','x']
    ci=['red','blue','green','#ED7D31']
    size=[40,40,40,40]
    linewidths=[1,1,4,4]
    labels = ['Unlearn-Forget Set', 'Unlearn-Retain Set', 'Original-Forget Set', 'Original-Retain Set']
    # labels = ['', '', '', '']

    for i, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:pink']):
        ax.scatter(Y[i*len(unlearn_forget_label):(i+1)*len(unlearn_forget_label), 0], Y[i*len(unlearn_forget_label):(i+1)*len(unlearn_forget_label), 1], \
                s=size[i]*4.0, marker=mi[i], color=ci[i], label=labels[i], edgecolors='none', alpha=1.0, linewidths=linewidths[i])
    ax.legend()
    ax.grid(True)

    plt.title('(a) Diffusion Models')
    plt.tight_layout()

    plt.savefig('dm_mask_ratio_diff_tsne.png')
    plt.savefig('dm_mask_ratio_diff_tsne.pdf')


