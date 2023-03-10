from glob import glob
import numpy as np
import torch
from wisp.offline_renderer import OfflineRenderer
from PIL import Image
from tqdm import tqdm
from configuration import *


def get_run_name() -> str:
    path = os.path.join(RUNS_LOG_PATH, '*')
    a = glob(path)
    recent = a[-1].split("\\")[-1]
    return recent


def export_gif_video(pipeline, num_angles=180, camera_distance=4):
    renderer = OfflineRenderer(render_res=[1024, 1024], render_batch=4000)
    num_lods = 16
    camera_origin = [-3, 0.65, -3]
    angles = np.pi * 0.1 * np.array(list(range(num_angles + 1)))
    x = -camera_distance * np.sin(angles)
    y = camera_origin[1]
    z = -camera_distance * np.cos(angles)
    for d in range(num_lods):
        out_rgb = []
        for idx in tqdm(range(num_angles + 1), desc=f"Generating 360 Degree of View for LOD {d}"):
            out = renderer.shade_images(
                pipeline,
                f=[x[idx], y, z[idx]],
                t=[0, 0, 0],
                fov=30,
                lod_idx=d
            )
            out = out.image().byte().numpy_dict()
            if out.get('rgb') is not None:
                out_rgb.append(Image.fromarray(np.moveaxis(out['rgb'].T, 0, -1)))
        rgb_gif = out_rgb[0]
        gif_path = os.path.join(FBUFFERDUMP_PATH, "rgb.gif")
        rgb_gif.save(gif_path, save_all=True, append_images=out_rgb[1:], optimize=False, loop=0)


def load_pipeline(path, device):
    # should load to gpu by default
    # https://pytorch.org/docs/stable/generated/torch.load.html
    pipeline = torch.load(path)
    pipeline.eval()
    return pipeline


def generate_video():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = get_run_name()
    path = os.path.join(RUNS_LOG_PATH, run_name, "model.pth", )
    pipeline = load_pipeline(path=path, device=device)
    export_gif_video(pipeline=pipeline)
    print("Done Generating Video\n\n\n\n")


if __name__ == '__main__':
    print("Load and Show")
    generate_video()
    print("Done")
