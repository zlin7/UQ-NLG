import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans


def get_affinity_mat(logits, mode='disagreement', temp=None, symmetric=True):
    if mode == 'jaccard':
        return logits
    # can be weigheted
    if mode == 'disagreement':
        logits = (logits + logits.permute(1,0,2))/2
        W = logits.argmax(-1) != 0
    if mode == 'disagreement_w':
        W = torch.softmax(logits/temp, dim=-1)[:, :, 0]
        if symmetric:
            W = (W + W.permute(1,0))/2
        W = 1 - W
    if mode == 'agreement':
        logits = (logits + logits.permute(1,0,2))/2
        W = logits.argmax(-1) == 2
    if mode == 'agreement_w':
        W = torch.softmax(logits/temp, dim=-1)[:, :, 2]
        if symmetric:
            W = (W + W.permute(1,0))/2
    if mode == 'gal':
        W = logits.argmax(-1)
        _map = {i:i for i in range(len(W))}
        for i in range(len(W)):
            for j in range(i+1, len(W)):
                if min(W[i,j], W[j,i]) > 0:
                    _map[j] = _map[i]
        W = torch.zeros_like(W)
        for i in range(len(W)):
            W[i, _map[i]] = W[_map[i], i] = 1
    W = W.cpu().numpy()
    W[np.arange(len(W)), np.arange(len(W))] = 1
    W = W.astype(np.float32)
    return W

def get_D_mat(W):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    return D

def get_L_mat(W, symmetric=True):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    if symmetric:
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    else:
        raise NotImplementedError()
        # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
        L = np.linalg.inv(D) @ (D - W)
    return L.copy()

def get_eig(L, thres=None, eps=None):
    # This function assumes L is symmetric
    # compute the eigenvalues and eigenvectors of the laplacian matrix
    if eps is not None:
        L = (1-eps) * L + eps * np.eye(len(L))
    eigvals, eigvecs = np.linalg.eigh(L)

    #eigvals, eigvecs = np.linalg.eig(L)
    #assert np.max(np.abs(eigvals.imag)) < 1e-5
    #eigvals = eigvals.real
    #idx = eigvals.argsort()
    #eigvals = eigvals[idx]
    #eigvecs = eigvecs[:,idx]

    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    return eigvals, eigvecs

def find_equidist(P, eps=1e-4):
    from scipy.linalg import eig
    P = P / P.sum(1)[:, None]
    P = (1-eps) * P + eps * np.eye(len(P))
    assert np.abs(P.sum(1)-1).max() < 1e-3
    w, vl, _ = eig(P, left=True)
    #assert np.max(np.abs(w.imag)) < 1e-5
    w = w.real
    idx = w.argsort()
    w = w[idx]
    vl = vl[:, idx]
    assert np.max(vl[:, -1].imag) < 1e-5
    return vl[:, -1].real / vl[:, -1].real.sum()

class SpetralClusteringFromLogits:
    def __init__(self,
                 affinity_mode='disagreement_w',
                 eigv_threshold=0.9,
                 cluster=True,
                 temperature=3., adjust=False) -> None:
        self.affinity_mode = affinity_mode
        self.eigv_threshold = eigv_threshold
        self.rs = 0
        self.cluster = cluster
        self.temperature = temperature
        self.adjust = adjust
        if affinity_mode == 'jaccard':
            assert self.temperature is None

    def get_laplacian(self, logits):
        W = get_affinity_mat(logits, mode=self.affinity_mode, temp=self.temperature)
        L = get_L_mat(W, symmetric=True)
        return L

    def get_eigvs(self, logits):
        L = self.get_laplacian(logits)
        return (1-get_eig(L)[0])

    def __call__(self, logits, cluster=None):
        if cluster is None: cluster = self.cluster
        L = self.get_laplacian(logits)
        if not cluster:
            return (1-get_eig(L)[0]).clip(0 if self.adjust else -1).sum()
        eigvals, eigvecs = get_eig(L, thres=self.eigv_threshold)
        k = eigvecs.shape[1]
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        return kmeans.labels_

    def clustered_entropy(self, logits):
        from scipy.stats import entropy
        labels = self(logits, cluster=True)
        P = torch.softmax(logits, dim=-1)[:, :, 2].cpu().numpy()
        pi = find_equidist(P)
        clustered_pi = pd.Series(pi).groupby(labels).sum().values
        return entropy(clustered_pi)

    def eig_entropy(self, logits):
        W = get_affinity_mat(logits, mode=self.affinity_mode, temp=self.temperature)
        L = get_L_mat(W, symmetric=True)
        eigs = get_eig(L, eps=1e-4)[0] / W.shape[0]
        return np.exp(- (eigs * np.nan_to_num(np.log(eigs))).sum())

    def proj(self, logits):
        W = get_affinity_mat(logits, mode=self.affinity_mode, temp=self.temperature)
        L = get_L_mat(W, symmetric=True)
        eigvals, eigvecs = get_eig(L, thres=self.eigv_threshold)
        return eigvecs

    def kmeans(self, eigvecs):
        k = eigvecs.shape[1]
        self.rs += 1
        kmeans = KMeans(n_clusters=k, random_state=self.rs, n_init='auto').fit(eigvecs)
        return kmeans.labels_

def umap_visualization(eigvecs, labels):
    # perform umap visualization on the eigenvectors
    import umap
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(eigvecs)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
    return embedding