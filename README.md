# WatchDog-Event-Trigger
watchdog event trigger on new file or folder creation / multi threaded multiple folder monitoring 

# config file contains all the directory paths 

    image_configurations:
        image_dump: main/images/imagedump   --> image dump directory path
        processed: main/images/processed    --> image processed directory
        results: main/images/results        --> image results saving directory
        csv_file: image_results.csv         --> csv file name

    same for the video and audio sections

    model_configurations:
        image_net_path: models/mobilenet_v2.pt   --> torch model destinations
        audio_net_path: models/mobilenet_v2.pt   
        video_net_path: models/model_name.pt


# main python script is directory_watcher.py  

    python3 directory_watcher.py

# algorithm fully modularized into classes for each image , audio , video easy incae of changing and understanding

# audio configuration such as model loading , pre processing goes same as image methods provided
# video section complete the read input video , processed save the processed and original videos into respective folders.

#  Thank you !!!

