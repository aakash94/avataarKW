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
* `pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html`
* Install FFMPEG
* Install Colman
* Make sure the paths are correct in `src/configuration.py`.
* create a `res` folder parallel to the `src` folder.
* follow the following folder structure, or change the paths accordingly
```
res/
|____dataset/
     |____transforms.json
     |____images/
|____framebuffer_dump/
     |____rgb.gif
|____colmandb/
|____imput.mp4
```

## Execution 
* Run `src/Main.py` to generate the dataset, perform Nerf, and generate a video from it. 
* `input.mp4` is the input video
*  `rgb.gif` is the final generated output


## Future Work
* Tweak hyper parameters to get better quality results