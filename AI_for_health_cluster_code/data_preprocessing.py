import numpy as np
import random
import h5py


def shuffle_channels_chance(ecg_heart_rate, p=0.5):
    """
    ecg_heart_rate: The heart rate data whose channels need to be shuffled
    p: The chance of a shuffle occurring
    
    returns: In 1st dimension shuffled data, or unshuffled data
    
    shuffles the four channels of the ECG heart rate data. 
    """
    if random.random() < p:
        np.random.shuffle(ecg_heart_rate)
    return ecg_heart_rate

def transform_channels_chance(ecg_heart_rate, p=0.1):
    """
    ecg_heart_rate: The heart rate data that needs to be transformed
    p: The chance of a transformation occurring
    returns: transformed data or untransformed data
    
    transforms data with a random scalar value.     
    """
    if random.random() < p:
        scalar = random.uniform(0.5, 1.3)
        return scalar * np.array(ecg_heart_rate)
    return ecg_heart_rate

def create_twin_heart_rate(ECG_transformed_1, ECG_transformed_2):
    """
    ECG_transformed_1: Single heart rate 1
    ECG_transformed_2: Single heart rate 2

    returns: twin heart rate
    
    Combines two single heart rates to create a twin heart rate.
    """
    return ECG_transformed_1 + ECG_transformed_2
    
def check_range_hr_nr(hr, missing_patients, exclude_patients):
    """
    hr: heart ecg number in dataset
    
    returns: the parient number, because some patients have been removed from the dataset the hr number in the dataset does not correspond to the patient number and the number needs to be changed
    """
    if hr < 10:
        return str('00' + str(int(hr)))
    elif hr > 99:
        if hr == missing_patients[1]:
            hr = hr + 1
        if hr == exclude_patients[1]:
            hr = hr + 1
        if hr == exclude_patients[2]:
            hr = hr + 1
        if hr == exclude_patients[3]:
            hr = hr + 1
        if hr == exclude_patients[4]:
            hr = hr + 1
        return str(int(hr))
    else:
        if hr == missing_patients[0]:
            hr = hr + 1
        if hr == exclude_patients[0]:
            hr = hr + 1
        return str('0' + str(int(hr)))
    
def take_random_snippet(hr, length_data, input_size, num_channels):
    """
    hr: the foetal heart ECG
    length_data: model_config["length_data"]
    input_size:  model_config["input_size"]
    num_channels: model_config["num_channels"]
    
    returns: a random snippet from the ECG data of length input_size
    """
    start_x = random.randint(0, (length_data - input_size))
    #print('startx', start_x)
    hr = np.delete(hr, np.s_[start_x:start_x + input_size], 1)
    return np.resize(hr,(num_channels, input_size))
    
def take_snippet_at_pos(hr, length_data, input_size, num_channels):
    """
    hr: the foetal heart ECG
    length_data: model_config["length_data"]
    input_size:  model_config["input_size"]
    num_channels: model_config["num_channels"]
    
    returns: the middle snippet from the ECG data of length input_size
    """
    start_x = length_data // 2 - input_size // 2
    hr = np.delete(hr, np.s_[start_x:start_x + input_size], 1)
    return np.resize(hr,(num_channels, input_size))

def generate_data(model_config, output_shape_1):
    """
    model_config: model configurations
    output_shape_1: the length of the output of the model
    
    returns: x_train, y_train1, y_train2, x_test, y_test1, y_test2, hr_1_list , hr_2_list
    """
    x_train = np.zeros((model_config["data_points_train"], model_config["input_size"], model_config["num_channels"]))
    y_train = np.zeros((model_config["data_points_train"], output_shape_1, model_config["num_channels"]*model_config["number_extractions"]))
    x_test_long = []
    y_test_long = []
    
    # length of data is total length minus excluded 
    cropped_length_data = model_config["length_data_points"] - len(model_config["exclude_patients"])
    
    # generate test data heart ECG numbers
    random_test_data_hr_1 = random.sample(range(0, cropped_length_data), model_config["data_points_test"])
    random_test_data_hr_2 = random.sample(range(0, cropped_length_data), model_config["data_points_test"])
    
    hr_1_list = []
    hr_2_list = []
    
    with h5py.File(model_config["hdf_ECG_file"], 'r') as f:
        singleton_key = list(f.keys())[0]

        # Get the data
        data = np.array(f[singleton_key])
        
        # get training data
        for i in range(model_config["data_points_train"]): 
            # Get random heart ECG 1
            ECG_transformed_1 = data_selector(data, cropped_length_data, random_test_data_hr_1 + random_test_data_hr_2, 
                                             model_config["random_snippet"],
                                             model_config["length_data"], 
                                             model_config["input_size"],
                                             model_config["num_channels"])
            # Get random heart ECG 2
            ECG_transformed_2 = data_selector(data, cropped_length_data, random_test_data_hr_1 + random_test_data_hr_2, 
                                             model_config["random_snippet"],
                                             model_config["length_data"], 
                                             model_config["input_size"],
                                             model_config["num_channels"])

            ECG_transformed_1 = np.clip(ECG_transformed_1, -2, 2)
            ECG_transformed_2 = np.clip(ECG_transformed_2, -2, 2)
            
            # Transform the data to make it more realistic
            ECG_transformed_1 = transform_channels_chance(shuffle_channels_chance(ECG_transformed_1, model_config["chuffle_chance"]), model_config["transform_chance"])
            ECG_transformed_2 = transform_channels_chance(shuffle_channels_chance(ECG_transformed_2, model_config["chuffle_chance"]), model_config["transform_chance"])

            
            ECG_transformed_1 = normalizer(ECG_transformed_1)
            ECG_transformed_2 = normalizer(ECG_transformed_2)
            # Add up data to get combination of two heart rates
            twin_ecg_heart = create_twin_heart_rate(ECG_transformed_1, ECG_transformed_2)

            x_train[i,:,:] = twin_ecg_heart.T
            y_train[i,:,4:] = ECG_transformed_1[:,:output_shape_1].T # Y in the model.fit
            y_train[i,:,:4] = ECG_transformed_2[:,:output_shape_1].T # Y in the model.fit

        # get testing data
        for i in range(model_config["data_points_test"]): 
            # Get random heart ECG 1
            hr_1 = random_test_data_hr_1[i]
            # Get random heart ECG 2
            hr_2 = random_test_data_hr_2[i]
            
            # Misschien ipv nieuwe list maken, de al bestaande random_test_data_hr_1 aanpassen 
            # Track heart numbers for testing
            hr_1_list.append(check_range_hr_nr(hr_1, model_config["missing_patients"], model_config["exclude_patients"]))
            hr_2_list.append(check_range_hr_nr(hr_2, model_config["missing_patients"], model_config["exclude_patients"]))
            
            #print(check_range_hr_nr(hr_1, model_config["missing_patients"], model_config["exclude_patients"]))
            
            ECG_transformed_1 = np.clip(data[hr_1], -2, 2)
            ECG_transformed_2 = np.clip(data[hr_2], -2, 2)
            
            length_test_data = min([ECG_transformed_1.shape[1], ECG_transformed_2.shape[1]])
            
            # Transform the data to make it more realistic
            ECG_transformed_1 = transform_channels_chance(shuffle_channels_chance(ECG_transformed_1, model_config["chuffle_chance"]), model_config["transform_chance"])
            ECG_transformed_2 = transform_channels_chance(shuffle_channels_chance(ECG_transformed_2, model_config["chuffle_chance"]), model_config["transform_chance"])

            # Add up data to get combination of two heart rates
            twin_ecg_heart = create_twin_heart_rate(ECG_transformed_1, ECG_transformed_2)
            

            ECG_transformed = np.zeros((length_test_data,
                                        model_config["num_channels"]*model_config["number_extractions"]))
            ECG_transformed[:,4:] = ECG_transformed_1.T
            ECG_transformed[:,:4] = ECG_transformed_2.T
            
            ECG_cut_up = []
            twin_cut_up = []
            
            input_index = 0
            output_index = 0
            
            while input_index < len(ECG_transformed)-model_config["input_size"]:
                twin_cut_up.append(twin_ecg_heart.T[input_index:input_index+model_config["input_size"]])
                ECG_cut_up.append(ECG_transformed[output_index:output_index+output_shape_1])

                output_index += output_shape_1
                input_index += model_config["input_size"]
  
            x_test_long.append(twin_cut_up)
            y_test_long.append(ECG_cut_up)
    return x_train, y_train, np.array(x_test_long), np.array(y_test_long), hr_1_list , hr_2_list

def data_selector(data, cropped_length_data, random_test_data, random_snippet, length_data, input_size, num_channels):
    """
    data: heart ECG data
    cropped_length_data: nr. datapoints with unusable datapoints removed
    random_test_data: random_test_data_hr_1 + random_test_data_hr_2
    
    returns: the heart ecg number to be used in the model training
    
    checks whether the ecg measurement to be added to the training data is in the test set and whether the data is a valid measurement and not more than 30% 0's
    """   
    while True:
        hr = random.randrange(0, cropped_length_data)
        if hr not in random_test_data:
            ECG_transformed = data[hr]
            # Take snippet from data
            if input_size == length_data:
                pass
            elif random_snippet:
                ECG_transformed = take_random_snippet(ECG_transformed, length_data, input_size, num_channels)
            else:
                ECG_transformed = take_snippet_at_pos(ECG_transformed, length_data, input_size, num_channels)
            # if less than 30% of data is 0's, accept the data
            if np.count_nonzero(ECG_transformed==0)/(len(ECG_transformed[0])*len(ECG_transformed)) < 0.3:
                return ECG_transformed
                    

def normalizer(data):
    """
    data: the data to be normalized
    
    returns: the normalized data
    """
    max_val = data.max()
    min_val = data.min()
    data = 2 * ((data - min_val) / (max_val - min_val)) - 1
    return data

