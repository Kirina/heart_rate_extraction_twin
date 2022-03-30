import matplotlib.pyplot as plt
import os

def plot_heartrate(ecg_heart_rate, data_number=0):
    """
    ecg_heart_rate: The data to be plotted
    data_number: The number of the data in the plot
    
    plots ECG heart rate data in a pyplot
    """
    plt.figure(figsize=(15,5))
    plt.plot(ecg_heart_rate)
    plt.title('Heart ECG plot number: ' + str(data_number))
    plt.ylim(-4, 4)
    plt.show()

def plot_loss(history):
    """
    history: model history
    
    plots the loss history of the model in a pyplot
    """
    plt.plot(history.history['loss'], label='loss')
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
        if not os.path.exists(output_path+'/history'):
            os.makedirs(output_path+'/history')
            os.makedirs(output_path+'/model')
            os.makedirs(output_path+'/predictions')
            os.makedirs(output_path+'/test_predictions')
            os.makedirs(output_path+'/train_data')
        print('output directories created')