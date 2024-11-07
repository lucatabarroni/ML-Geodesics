'''
Copyright (C) November 2024  Alessandro De Santis, alessandro.desantis@roma2.infn.it
'''


import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import time
import numpy as np
import matplotlib.pyplot as plt
import sys
  
from geometry   import PlanarGeometry, NonPlanarGeometry, SeparableGeometry, NonlinearOscillator, ChaoticOscillator
from pathlib    import Path
from service    import load_config, Settings, missing_key_error, methods
from service_ML import Training 
from optparse   import OptionParser
from tensorflow import keras


# from google.colab import drive
# drive.mount('/content/drive')

parser = OptionParser()
parser.add_option( "-i" , "--inputfile" , dest="inputfile", default= "input.yaml")
(options, args)  = parser.parse_args()
input_dictionary = load_config(Path(options.inputfile))

### initialization ############

input  = Settings(input_dictionary)
keras.utils.set_random_seed(input.seed)

if input.geometry_type == 'planar':
    geometry     = PlanarGeometry(input_dictionary["geometry"]) if "geometry"  in input_dictionary.keys() else missing_key_error("geometry")

if input.geometry_type == 'separable':
    geometry     = SeparableGeometry(input_dictionary["geometry"]) if "geometry"  in input_dictionary.keys() else missing_key_error("geometry")

if input.geometry_type == 'non_planar':
    geometry     = NonPlanarGeometry(input_dictionary["geometry"]) if "geometry"  in input_dictionary.keys() else missing_key_error("geometry")

if input.geometry_type == 'nonlinear_oscillator':
    geometry     = NonlinearOscillator(input_dictionary["geometry"]) if "geometry"  in input_dictionary.keys() else missing_key_error("geometry")

if input.geometry_type == 'chaotic_oscillator':
    geometry     = ChaoticOscillator(input_dictionary["geometry"]) if "geometry"  in input_dictionary.keys() else missing_key_error("geometry")





training   = Training(input_dictionary["training"],input,geometry) if "training"  in input_dictionary.keys() else missing_key_error("training")
training.set_model(input,geometry)
geometry.gnuplot_scripts(input,training,methods)

#### training and prediction ###

if input.mode == 'training' or input.mode == 'restart':
    training.write_info(input,geometry)
    training.perform_training(input)

training.prediction(input,geometry,methods)

print('done')

del input, geometry, training, input_dictionary









