import matplotlib.pyplot as plt
import os
import numpy as np

def plot_heartecg(ecg_heart_rate, data_number=0):
    """
    ecg_heart_rate: The data to be plotted
    data_number: The number of the data in the plot
    
    plots ECG heart rate data in a pyplot
    """
    plt.figure(figsize=(15,5))
    plt.plot(ecg_heart_rate)
    plt.title('Heart ECG plot number: ' + str(data_number))
    plt.ylim(-4, 4)
#     plt.xlim(-1.1, 1.1)
    plt.show()

def plot_loss(history):
    """
    history: model history
    
    plots the loss history of the model in a pyplot
    """
    plt.plot(history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)   
    
def create_directory(output_path):
    """
    output_path: the path from source to the output folder 
    
    creates the folders needed to store the output data
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path+'/history')
        os.makedirs(output_path+'/model')
        os.makedirs(output_path+'/predictions')
        os.makedirs(output_path+'/test_predictions')
        os.makedirs(output_path+'/train_data')
        os.makedirs(output_path+'/BSS')
        
        print('output directories created')
        
        
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
    
def real_ecg_with_scatter(mat_file, S_twin_ecg, first, second, title):
    """
    mat_file: MatLab data file with foetal data
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
        