# avataarKW


## Task
Use your smartphone to capture 360 degrees video of an object.
The task is to make use of the Kaolin Wisp library to create a new trajectory video of the
object.
The output can look something like this:
https://drive.google.com/file/d/1ZFfX2LvF8lxGWrrLnA4dhqBto62D91PE/view?usp=share_link

In this video, the rendered camera trajectory is different from the captured trajectory.
You can play with speed, zoom, focal length as well.

You have to submit the github link of your code.



## Setup
* Install [Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/INSTALL.md#installation-steps-in-detail)
* Install FFMPEG
* Install Colman
* Make sure the paths are correct in `src/configuration.py`.

## Execution 
* Run `src/Main.py` to generate the dataset
* Run `python src/nerf_main.py --config src/nerf_configs/nerf_octree.yaml --dataset-path /res/dataset` to render the scene in 3d and use the tool to generate images for video


## Future Work
* Tweak hyper parameters to get better quality results
* Simpler script to auto generate video