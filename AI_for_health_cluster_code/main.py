import tensorflow as tf
import numpy as np
import keras
import h5py

from unet_config import config
from data_preprocessing import generate_data
from utils import create_directory
from model import wave_u_net

model_config = config()
# Parameters for the Wave-U-net

params = {
  "num_initial_filters": model_config["num_initial_filters"],
  "num_layers": model_config["num_layers"],
  "kernel_size": model_config["kernel_size"],
  "merge_filter_size": model_config["merge_filter_size"],
  "source_names": model_config["source_names"],
  "num_channels": model_config["num_channels"],
  "num_channels_output": model_config["num_channels_output"],
  "output_filter_size": model_config["output_filter_size"],
  "padding": model_config["padding"],
  "input_size": model_config["input_size"],
  "context": model_config["context"],
  "upsampling_type": model_config["upsampling_type"],         # "learned" or "linear"
  "output_activation": model_config["output_activation"],        # "linear" or "tanh"
  "output_type": model_config["output_type"],          # "direct" or "difference"
  "dropout_rate": model_config["dropout_rate"],
}

m = wave_u_net(**params)

output_shape = (model_config["batch_size"], m.output[model_config["source_names"][0]].shape[1], model_config["num_channels"]*2)
input_shape = (model_config["batch_size"], model_config["input_size"], model_config["num_channels"])

print('before data creation')

x_train, y_train, x_test_long, y_test_long, hr_1_list, hr_2_list = generate_data(model_config, output_shape[1])

print('created train data etc')

loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
opt = tf.keras.optimizers.Adam(learning_rate=model_config["init_sup_sep_lr"])
m.compile(optimizer=opt, loss=loss_fn, metrics=["mae"])


print('start fit ')
history = m.fit(x=x_train, y=y_train, validation_split=model_config["validation_split"], batch_size=model_config["batch_size"], epochs=model_config["epochs"])

create_directory(model_config["output_path"])

m.save_weights(model_config["output_path"]+'/model'+'/saved_model_weights')

np.save(model_config["output_path"]+'/history/history1.npy',history.history)

predictions_list = []

for entries in range(len(x_test_long)):
    print(len(x_test_long[entries]))
    
    prediction = m.predict(np.array(x_test_long[entries]))
    predictions_list.append(prediction)
    
np.save(model_config["output_path"]+'/test_predictions/hr_1_list.npy',hr_1_list)
np.save(model_config["output_path"]+'/test_predictions/hr_2_list.npy',hr_2_list)

np.save(model_config["output_path"]+'/train_data/x_train.npy', x_train)
np.save(model_config["output_path"]+'/train_data/y_train.npy', y_train)

for i, predictions in enumerate(predictions_list):
    
    dim_1, dim_2, _ = predictions[model_config["source_names"][0]].shape
    
    y_test1_single_all = np.zeros((1, dim_1*dim_2, model_config["num_channels"]))
    y_test2_single_all = np.zeros((1, dim_1*dim_2, model_config["num_channels"]))
    y_pred_test1_single_all = np.zeros((1, dim_1*dim_2, model_config["num_channels"]))
    y_pred_test2_single_all = np.zeros((1, dim_1*dim_2, model_config["num_channels"]))
    x_test_single_all = np.zeros((1, dim_1*model_config["input_size"], model_config["num_channels"]))
    
    for sample in range(dim_1):
        # get number of heart ecgs
        hr_1 = hr_1_list[i]
        hr_2 = hr_2_list[i]
        
        # Get x and y values 
        y_test1_single = y_test_long[i, sample, :, model_config["num_channels"]:][None, ...]
        y_test2_single = y_test_long[i, sample, :, :model_config["num_channels"]][None, ...]
        x_test_single = x_test_long[i, sample, :, :][None, ...]
        
        predictions_test = m.predict(x_test_single)
        
        pred_test = predictions_test[model_config["source_names"][0]]
        
        y_test1_single_all[:,sample*dim_2:(sample+1)*dim_2,:] = y_test1_single
        y_test2_single_all[:,sample*dim_2:(sample+1)*dim_2,:] = y_test2_single
        y_pred_test1_single_all[:,sample*dim_2:(sample+1)*dim_2,:] = pred_test[:,:,model_config["num_channels"]:]
        y_pred_test2_single_all[:,sample*dim_2:(sample+1)*dim_2,:] = pred_test[:,:,:model_config["num_channels"]]
        x_test_single_all[:,sample*model_config["input_size"]:(sample+1)*model_config["input_size"],:] = x_test_single
    
    with h5py.File(model_config["output_path"]+f'/test_predictions/patient_{hr_1}{hr_2}.signal', 'w') as f:
      f.create_dataset('hr_1_original', data=y_test1_single_all)
      f.create_dataset('hr_2_original', data=y_test2_single_all)
      f.create_dataset('hr_1_extracted', data=y_pred_test1_single_all)
      f.create_dataset('hr_2_extracted', data=y_pred_test2_single_all)
      f.create_dataset('twin_hr', data=x_test_single_all)