'''
Copyright (C) November 2024  Alessandro De Santis, alessandro.desantis@roma2.infn.it
'''

import os 
import numpy      as np
from   pathlib    import Path
from   typing     import Any, List
from   scipy      import special
from   service_ML import Training
import shutil
import struct
import yaml

methods = ['BDF','RK23','RK45','Radau']

def log(obj,verbose: int = 1):
  if verbose == 1:
    print(obj)

def NewDirectory(directory_path: Path,option: str = 'empty_dir') -> None:
    
  if option =='empty_dir':  # remove directory if already existing
  
    if directory_path.exists() and directory_path.is_dir():
      print('Already existing directory -> remove')
      shutil.rmtree(directory_path)
  
  if directory_path.exists() == False:
    directory_path.mkdir(parents=True, exist_ok=True)
    print(f'Create destination directory at {directory_path}')


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)    
    return config


def missing_key_error(key):
    raise Exception(f'<{key}> missing')


class Settings:
   
    def __init__(self, dict: Any,*args,**kargs):
        
        print('Read input file ...\n\n')

        self.seed             = dict["keras_seed"]             if "keras_seed"       in dict.keys() else 217
        self.extra_tag        = dict["extra_tag"]              if "extra_tag"        in dict.keys() else ""
        self.mode             = dict["mode"]                   if "mode"             in dict.keys() else "training"
        self.geometry_type    = dict["geometry_type"]          if "geometry_type"    in dict.keys() else "geometry_type"
        self.destination_path = Path(dict["destination_path"]) if "destination_path" in dict.keys() else Path("results")

    def create_output_directory(self,Training,geometry):


      if ('oscillator' in self.geometry_type) == False: 
        additional_name = Path(f'{self.geometry_type}_b{geometry.b}_{Training.architecture_name}_lri{Training.lri}_lambda{Training.lambda_}_scheduler{Training.lr_scheduler_type}_Npoints{len(Training.training_time)}{self.extra_tag}')
      else:
        additional_name = Path(f'{self.geometry_type}_{Training.architecture_name}_lri{Training.lri}_lambda{Training.lambda_}_scheduler{Training.lr_scheduler_type}{self.extra_tag}')
          
  
      self.destination_path = self.destination_path / additional_name

      if (self.destination_path).exists() == False:
        NewDirectory(self.destination_path,'empty_dir')
       

