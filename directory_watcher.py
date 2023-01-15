# import time module, Observer, FileSystemEventHandler
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# importing the multiprocessing module
import multiprocessing
from config_parser import conf
from image_utils import Image_Model
from audio_utils import Audio_Model
from video_utils import Video_Model
import cv2
import os
import gc

# define 3 classes for each image , audio , video
class_dict ={
    "image" : Image_Model() ,
    "audio" : Audio_Model() ,
    "video" : Video_Model()
}


class Handler(FileSystemEventHandler):

    def __init__(self , type) -> None:
        super().__init__()
        self.type = type
        self.function = class_dict[ type ]


    def on_any_event(self,event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            
                if( self.type != 'video' ):
                    # Event is created, you can process it now
                    print("Watchdog received created event - % s." % event.src_path)
                    print("Invoke function type ", self.type )
                    time.sleep(5)
                    # invoke the function to predict based on the image
                    self.function.predict( event.src_path )

                    # dump the image into the procced folder
                    self.function.save_processed( event.src_path )
                
                else:
                    # Event is created, you can process it now
                    print("Watchdog received created event - % s." % event.src_path)
                    print("Invoke function type ", self.type )
                    time.sleep(5)
                    # invoke the function to predict based on the image
                    self.function.vid_resize( event.src_path , 250 )

                    # dump the image into the procced folder
                    self.function.save_original( event.src_path )


        elif event.event_type == 'modified':
            # Event is modified, you can process it now
            print("Watchdog received modified event - % s." % event.src_path)

def monitor_folder(watchDirectory , type):

    observer = Observer()
    # Set the directory on watch
    watchDirectory = watchDirectory

    event_handler = Handler(type)
    observer.schedule(event_handler, watchDirectory, recursive = True)
    observer.start()
    try:
        while True:
            time.sleep(5)
    except:
        observer.stop()
        print("Observer Stopped")

    observer.join()


def startProcess():
    ''' Uses multiprocessing module to start 3 different threads for each directory'''
    # get the dump directories from the config file
    jobs=[ conf.audio_configurations.audio_dump , conf.image_configurations.image_dump , conf.video_configurations.video_dump ]
    # get processing types
    types =[ "audio" , "image" , "video"]
    threds =[]
    #pdb.set_trace()

    #log.info(mp.log_to_stderr(logging.DEBUG))
    for i in range(len(jobs)):

        try:
            # apply multiprocessing with watchdog for each directory , which runs paralle
            p = multiprocessing.Process(target= monitor_folder , args=( jobs[i] , types[i] ) )
            p.daemon=True
            threds.append(p)
            p.start()

        except KeyboardInterrupt:

            print("Threading Exception")

    for i in range(len(threds)):
        threds[i].join()  

    return   		

if __name__ == '__main__':
    

    startProcess()
