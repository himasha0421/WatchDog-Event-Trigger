import io
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import datetime
from config_parser import conf
import os
import cv2
import gc

class Audio_Model :

    def __init__(self) -> None:
        # initial model loading
        self.load_model()
        # target labels
        self.labels = ["output1", "output2", "output3", "output4" , "output5" ]
        self.csv_path = os.path.join( conf.audio_configurations.results , conf.audio_configurations.csv_file )
        self.processed_dir = conf.audio_configurations.processed

    def load_model( self ):

        try:
            # load the torch audio model
            self.torchscript_model = torch.jit.load( conf.model_configurations.audio_net_path ).eval()
            # audio preprocessing method ( currently using same audio method)
            torchscript_tfs = torch.nn.Sequential(
                T.Resize((256,256)),
                T.CenterCrop(224),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            )
            self.scripted_transforms = torch.jit.script(torchscript_tfs)

        except :
            print("Error Model Loading !!!")

    def predict( self , input_file ):
        if input_file:
            # current date and time

            try :
                timestamp = datetime.datetime.now()
                # audio file reading method ( currently using same image loading method )
                image = Image.open(  input_file ).convert('RGB')

                image_tensor = F.to_tensor(image).unsqueeze(0)
                preds = torch.softmax( self.torchscript_model(  self.scripted_transforms(image_tensor)) ,dim=1 )
                probs =[ preds[0][0].item() , preds[0][1].item() , preds[0][2].item() , preds[0][3].item() ]

                results_target = self.labels[ np.argmax(probs) ]

                self.save_results( file_path=input_file.split( os.sep )[-1] , timestamp=timestamp , pred_target= results_target  )

                # flush unwanted variables
                del image , image_tensor ; gc.collect();

            except:
                print("Model Prediction Error !!!!!")



    def save_results( self , file_path , timestamp , pred_target  ):

        try :
            # check the csv file path 
            if( os.path.exists( self.csv_path ) ) :

                # open the csv and append the row with 1. image name      2.Date     3. result
                df = pd.read_csv( self.csv_path )
                # append the data into data frame
                df = df.append( {'Name': file_path , "Date":timestamp , "Results":pred_target   }    , 
                                ignore_index=True)

                df.to_csv( self.csv_path ,  index=False)

            else:
                # define new data frame
                df = pd.DataFrame(columns=['Name' , 'Date' , 'Results' ])
                # append new row with results
                df = df.append( {'Name': file_path , "Date":timestamp , "Results":pred_target   }    , 
                                ignore_index=True)
                df.to_csv(  self.csv_path , index=False)

            # flush the df variable
            del df ; gc.collect();

        except:
            print("Results Saving Error !!!")

    def save_processed( self , input_file ):
        try:
            # after processed move the audio into processed folder ( using same method used in images need to change according to audios )
            image = cv2.imread( input_file  )
            save_path = os.path.join( self.processed_dir ,  input_file.split(os.sep)[-1]  )
            # save the image to processed folder ( using image mthod need to change according to audio )
            cv2.imwrite( save_path , image )

            # flush the image and invoke garbage collector
            del image ; gc.collect();
        except:
            print("Original Audio Save Error !!!!")


