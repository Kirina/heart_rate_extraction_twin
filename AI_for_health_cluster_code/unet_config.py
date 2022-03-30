"""
Model configurations for the wave-u-net
"""

def config():
    model_config = {"hdf_ECG_file" : "pregnancy_data_training_long.h5",
                    "cropped_singleton_ECG_dataset" : "cropped_singleton_ECG",
                    "hdf_ECG_file_predictions" : "model_predictions.h5",
                    "predictions_1" : "predictions_1",
                    "predictions_2" : "predictions_2",
#                     "cropped_twin_ECG_dataset" : "cropped_twin_ECG",
                    "data_path" : "data", # Set this to where the preprocessed dataset should be saved
                    "length_data_points" : 495,
                    "length_data": 147443, 
                    #change this to use more of the training data
                    "data_points_train" : 3000,                    
                    "data_points_test" : 10,
                    "missing_patients" : [63, 485], 
                    "exclude_patients" : [18, 103, 141, 151, 382], # Recording too short
                    "source_names" : ["heart_rates"],
                    "number_extractions" : 2,
                    "random_snippet": True,
                    "chuffle_chance" : 0.5,
                    "transform_chance" : 0.2,
                    "validation_split" : 0.2,
                    "init_sup_sep_lr" : 1e-5, # Supervised separator learning rate
                    "batch_size" : 10, # Batch size
                    "epochs" : 200,
                    "epoch_it" : 2000, # Number of supervised separator steps per epoch
                    "cache_size": 4000, # Number of audio snippets buffered in the random shuffle queue. Larger is better, since workers put multiple examples of one song into this queue. The number of different songs that is sampled from with each batch equals cache_size / num_snippets_per_track. Set as high as your RAM allows.
                    "num_layers" : 5, # How many U-Net layers (5)
                    "kernel_size": 15,
                    "num_channels": 4, #input channels model
                    "num_channels_output": 8, #output channels model
                    "merge_filter_size" : 5, # For Wave-U-Net: Filter size of conv in upsampling block
                    "output_filter_size": 1, # For Wave-U-Net: Filter size of convolution in the output layer
                    "num_initial_filters" : 24, # Number of filters for convolution in first layer of network
                    "padding": "valid",
                    "random_snippet": True,
#                     "input_size": 147443,
                    "input_size": 22050,
                    "dropout_rate": 0.5,
                    "expected_sr": 22050,  # Downsample all audio input to this sampling rate
                    "mono_downmix": True,  # Whether to downsample the audio input
#                     "output_type" : "difference", # Type of output layer, either "direct" or "difference". Direct output: Each source is result of tanh activation and independent. DIfference: Last source output is equal to mixture input - sum(all other sources)
                    "output_path":"/mnt/netcache/diag/kirina/output_model",
                    "output_type": "direct",
#                     "output_activation" : "linear", # Activation function for output layer. "tanh" or "linear". Linear output involves clipping to [-1,1] at test time, and might be more stable than tanh
                    "output_activation": "tanh",
                    "context" : False, # Type of padding for convolutions in separator. If False, feature maps double or half in dimensions after each convolution, and convolutions are padded with zeros ("same" padding). If True, convolution is only performed on the available mixture input, thus the output is smaller than the input
    #                 "network" : "unet", # Type of network architecture, either unet (our model) or unet_spectrogram (Jansson et al 2017 model)
#                     "upsampling_type" : "learned", # Type of technique used for upsampling the feature maps in a unet architecture, either "linear" interpolation or "learned" filling in of extra samples
                    "upsampling_type": "linear",
                    }
    return model_config