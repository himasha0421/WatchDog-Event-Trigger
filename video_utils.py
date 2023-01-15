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
import warnings
import ffmpeg
import gc

class Video_Model :

    def __init__(self) -> None:
        
        # apply if needed
        #self.load_model()
        self.csv_path = os.path.join( conf.video_configurations.results , conf.video_configurations.csv_file )
        self.ffmpeg_path = conf.video_configurations.ffmpeg
        self.video_processed = conf.video_configurations.processed

    def vid_resize( self , vid_path , width, overwrite = False):
        '''
        use ffmpeg to resize the input video to the width given, keeping aspect ratio
        '''
        try:
            # get the output path to save the processed video with ffmpeg 
            output_vid_name = "new_"+str( vid_path.split(os.sep)[-1].split(".")[0] )+".mp4"
            output_path = os.path.join( self.ffmpeg_path , output_vid_name )

            if not( os.path.isdir(os.path.dirname(output_path))):
                raise ValueError(f'output_path directory does not exists: {os.path.dirname(output_path)}')

            if os.path.isfile(output_path) and not overwrite:
                warnings.warn(f'{output_path} already exists but overwrite switch is False, nothing done.')
                return None
            # read input video
            input_vid = ffmpeg.input(vid_path)
            # video processing
            vid = (
                input_vid
                .filter('scale', width, -1)
                .output(output_path)
                .global_args('-loglevel', 'quiet')
                .global_args('-y')
                .overwrite_output()
                .run()
            )

            # flushed the unwanted variables
            del input_vid , vid ; gc.collect() ;

            return output_path
        
        except :
            print("Video Processing Error !!!!")

    def save_original( self , vid_path ):

        try :
            # get the input vido file
            save_filename = vid_path.split(os.sep)[-1]
            # get the final file path
            save_path = os.path.join( self.video_processed , save_filename )
            # read input video
            input_vid = ffmpeg.input( vid_path  )
            # save the original video into output folder without any change
            vid = (
                input_vid
                .output(save_path)
                .overwrite_output()
                .global_args('-loglevel', 'quiet')
                .global_args('-y')
                .run()
            )
            # flush the unwanted memory
            del input_vid , vid ; gc.collect() ;

        except:
            print("Original File Save Error !!!")

    