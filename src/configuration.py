from torch import cuda
import os
from math import floor
from subprocess import run, PIPE, STDOUT


# Main and configuration should be placed on same level

def get_length(filename):
    result = run(["ffprobe", "-v", "error", "-show_entries",
                  "format=duration", "-of",
                  "default=noprint_wrappers=1:nokey=1", filename],
                 stdout=PIPE,
                 stderr=STDOUT)
    return float(result.stdout)


def get_fps(filename):
    # because ideal number of frames is 50-150, so taking the number 150 here.
    duration = get_length(filename)
    fps = floor(150.0 / duration)
    return fps


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_PATH = os.path.join(ROOT_PATH, 'src')
RES_PATH = os.path.join(ROOT_PATH, 'res')
CONFIG_PATH = os.path.join(SRC_PATH, 'nerf_configs')
BASE_CONFIG_PATH = os.path.join(CONFIG_PATH, 'nerf_base.yaml')
HASH_CONFIG_PATH = os.path.join(CONFIG_PATH, 'nerf_hash.yaml')
OCTREE_CONFIG_PATH = os.path.join(CONFIG_PATH, 'nerf_octree.yaml')
TRIPLANAR_CONFIG_PATH = os.path.join(CONFIG_PATH, 'nerf_triplanar.yaml')
COLMAPDB_DIR_PATH = os.path.join(RES_PATH, 'colmapdb', '')
COLMAPDB_PATH = os.path.join(COLMAPDB_DIR_PATH, 'colmap.db')
# COLMAPTEXT_PATH = os.path.join(RES_PATH, 'colmap_text','')
TEXT_FOLDER_PATH = os.path.join(COLMAPDB_DIR_PATH, 'colmap_text', '')
IN_VIDEO_PATH = os.path.join(RES_PATH, 'input.mp4')
OUT_VIDEO_PATH = os.path.join(RES_PATH, 'output.mp4')
DATASET_PATH = os.path.join(RES_PATH, 'dataset')
IMAGES_PATH = os.path.join(DATASET_PATH, 'images', '')
JSON_PATH = os.path.join(DATASET_PATH, 'transforms.json')
COLMAP_PATH = os.path.join(SRC_PATH, 'colmap_nocuda')
SCALE_AABB = 16
if cuda.is_available():
    COLMAP_PATH = os.path.join(SRC_PATH, 'colmap_cuda')
COLMAPBAT_PATH = os.path.join(COLMAP_PATH, 'COLMAP.bat')
FPS_COUNT = get_fps(IN_VIDEO_PATH)

if __name__ == '__main__':
    print("Welcome to Configuration File")
    '''
    python C:/Users/Thominator/Desktop/InstantNGP/scripts/colmap2nerf.py --video_in video.MOV --video_fps 2 --run_colmap --aabb_scale 16
    --video_in video.MOV 
    --video_fps 2 
    --run_colmap 
    --aabb_scale 16
    '''
