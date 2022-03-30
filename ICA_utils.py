import numpy as np
import random
import matplotlib.pyplot as plt
from data_preprocessing import normalizer
from sklearn.decomposition import FastICA, PCA

def whiten(X):
    # Calculate the covariance matrix
    coVarM = covariance(X)

    # Single value decoposition
    U, S, V = np.linalg.svd(coVarM)

    # Calculate diagonal matrix of eigenvalues
    d = np.diag(1.0 / np.sqrt(S))

    # Calculate whitening matrix
    whiteM = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    Xw = np.dot(whiteM, X)

    return Xw, whiteM

def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n

def fastIca(signals, number_extractions, alpha = 1, thresh=1e-8, iterations=5000):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(number_extractions, m)

    for c in range(number_extractions):
            w = W[c, :].copy().reshape(m, 1)
            w = w / np.sqrt((w ** 2).sum())

            i = 0
            lim = 100
            while ((lim > thresh) & (i < iterations)):

                # Dot product of weight and signal
                ws = np.dot(w.T, signals)

                # Pass w*s into contrast function g
                wg = np.tanh(ws * alpha).T

                # Pass w*s into g prime
                wg_ = (1 - np.square(np.tanh(ws))) * alpha

                # Update weights
                wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

                # Decorrelate weights              
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())

                # Calculate limit condition
                lim = np.abs(np.abs((wNew * w).sum()) - 1)

                # Update weights
                w = wNew

                # Update counter
                i += 1

            W[c, :] = w.T
    return W

# def calculate_ica(twin_ecg, number_extractions):
#     Xw, whiteM = whiten(twin_ecg)
#     ica = fastIca(Xw, number_extractions, alpha=1)
#     #Un-mix signals using
#     S_twin_ecg = Xw.T.dot(ica.T)
#     return S_twin_ecg
    
def ICA_manual(twin_ecg, number_extractions):
    Xw, whiteM = whiten(twin_ecg)
    
    ica = fastIca(Xw, number_extractions, alpha=1)

    #Un-mix signals using
    S_twin_ecg = Xw.T.dot(ica.T)
#     S_twin_ecg = calculate_ica(twin_ecg, number_extractions)
    S_twin_ecg = normalizer(S_twin_ecg)
    
    return S_twin_ecg


def SKlearn_fast_ICA(twin_ecg):
    """    
    twin_ecg: Twin heart rate to be decoded
    
    Uses SKlearn ICA method to extract a heart rate 
    """
    ica = FastICA(n_components=2)
    S_twin_ecg = ica.fit_transform(twin_ecg.T)  # Reconstruct signals
    S_twin_ecg = normalizer(S_twin_ecg)
    # Plot the results
    return S_twin_ecg    
    
def SKlearn_PCA(twin_ecg):
    """    
    twin_ecg: Twin heart rate to be decoded
    
    Uses SKlearn ICA method to extract a heart rate 
    """
    pca = PCA(n_components=2)
    S_twin_ecg = pca.fit_transform(twin_ecg.T)  # Reconstruct signals
    S_twin_ecg = normalizer(S_twin_ecg)
    # Plot the results
    return S_twin_ecg  

def heart_ecg_separated_plot(S_twin_ecg, twin_ecg, twin_1, twin_2, title=''):
    """
    S_twin_ecg: Extracted twin heart rates
    twin_ecg: Twin heart rate to be decoded
    twin_1: Source 1
    twin_2: Source 2
    
    Plot heart rates extracted by SKlearn
    """
    fig = plt.gcf()
    fig.set_size_inches(30, 10)
    plt.plot(twin_ecg.T)
    plt.title('Mixed signal')
    plt.show()

    fig, ax = plt.subplots(2, figsize=[30, 10], sharex=True)
#     fig, ax = plt.subplots(2)
    ax[0].plot(twin_1.T, lw=1)
    ax[0].set_title('Original signal 1', fontsize=25)
    ax[0].tick_params(labelsize=12)
    
    print(S_twin_ecg.shape)
    ax[1].plot(S_twin_ecg[:,1], lw=1)
#     ax[1].plot(S_twin_ecg, lw=1)
    ax[1].tick_params(labelsize=12)
    ax[1].set_title('Extracted signal 1 ' + title, fontsize=25)
    #ax[1].set_xlim(ns[0], ns[-1])
    plt.show()

    fig1, ax1 = plt.subplots(2, figsize=[30, 10], sharex=True)
    ax1[0].plot(twin_2.T, lw=1)
    ax1[0].set_title('Original signal 2', fontsize=25)
    ax1[0].tick_params(labelsize=12)

    ax1[1].plot(S_twin_ecg[:,0], lw=1)
#     ax1[1].plot(S_twin_ecg, lw=1)
    ax1[1].tick_params(labelsize=12)
    ax1[1].set_title('Extracted signal 2 ' + title, fontsize=25)
    
    plt.show()


def real_ecg_separated_plots(S_twin_ecg, title):
    """
    S_twin_ecg: separated ecgs to be plotted
    title: title of plot
    
    Plots the ECG data of a real recording, so without a reference plot
    """
    fig, ax = plt.subplots(2, figsize=[30,10], sharex=True)
    ax[0].plot(S_twin_ecg[:,0], lw=1)
    ax[0].tick_params(labelsize=12)
    ax[0].set_title('Extracted signal 1 ' + title, fontsize=25)

    ax[1].plot(S_twin_ecg[:,1], lw=1)
    ax[1].tick_params(labelsize=12)
    ax[1].set_title('Extracted signal 2 ' + title, fontsize=25)

    plt.show()
    
def real_ecg_with_scatter(S_twin_ecg, first, second, title):
    """
    S_twin_ecg: separated ecgs to be plotted
    first: define whether ecg 1 of ecg 2 is plotted first
    second: define whether ecg 1 or 2 is plotted second
    title: title of plot
    
    Plots the ECG data of a real recording, so without a reference plot, includes a scatterplot of the extracted ECG peaks
    """
    y_fet1 = []
    for i in mat_file['rPeaks_fetus'+first].T:
        y_fet1.append(S_twin_ecg[:,0].T[i])

    fig = plt.gcf()
    fig.set_size_inches(30, 10)
    plt.title('Extracted signal 1 ' + title, fontsize=25)
    plt.ylim(-1.1, 1.1)
    plt.plot(S_twin_ecg[:,0], zorder=1)
    plt.scatter(mat_file['rPeaks_fetus'+first].T, y_fet1, s=5, zorder=2, c='red')
    plt.show()
    
    y_fet1 = []
    for i in mat_file['rPeaks_fetus'+second].T:
        y_fet1.append(S_twin_ecg[:,1].T[i])
    
    fig = plt.gcf()
    fig.set_size_inches(30, 10)
    plt.title('Extracted signal 2 ' + title, fontsize=25)
    plt.ylim(-1.1, 1.1)
    plt.plot(S_twin_ecg[:,1], zorder=1)
    plt.scatter(mat_file['rPeaks_fetus'+second].T, y_fet1, s=5, zorder=2, c='red')
    plt.show()
    
def plot_twin_peaks(S_twin_ecg, x_axis, y_axis):    
    """
    S_twin_ecg: separated ecgs to be plotted
    x_axis: data to be plotted on x axis
    y_axis: data to be plotted on y axis
    
    Plots extracted single ECG with ECG peaks annotated
    """
    y_fet = []

    for i in x_axis:
        y_fet.append(y_axis[i])

    fig = plt.gcf()
    fig.set_size_inches(30, 10)
    plt.ylim(-1.1, 1.1)

    plt.plot(S_twin_ecg, zorder=1)
    plt.scatter(x_axis, y_fet, s=5, zorder=2)

    plt.show()
    
def plot_both_twin_peaks(x_axis, y_axis):
    """
    x_axis: data to be plotted on x axis
    y_axis: data to be plotted on y axis
    
    Plots twin ECG with ECG peaks annotated
    """
    y_fet = []
    for i in y_axis:
        y_fet.append(x_axis[:1].T[i])

    fig = plt.gcf()
    fig.set_size_inches(30, 10)
    plt.ylim(-20, 20)

    plt.plot(np.clip(x_axis.T, -20, 20), zorder=1)
    plt.scatter(y_axis, y_fet, s=5, zorder=2)

    plt.show()