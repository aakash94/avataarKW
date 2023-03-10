import os
import argparse
import logging
import numpy as np
import torch
import wisp
from wisp.app_utils import default_log_setup, args_to_log_format
import wisp.config_parser as config_parser
from wisp.framework import WispState
from wisp.datasets import MultiviewDataset, SampleRays
from wisp.models.grids import BLASGrid, OctreeGrid, CodebookOctreeGrid, TriplanarGrid, HashGrid
from wisp.tracers import BaseTracer, PackedRFTracer
from wisp.models.nefs import BaseNeuralField, NeuralRadianceField
from wisp.models.pipeline import Pipeline
from wisp.trainers import BaseTrainer, MultiviewTrainer
from wisp.renderer.app.wisp_app import WispApp
from configuration import FBUFFERDUMP_PATH
from wisp.core import Rays, RenderBuffer
import wisp.ops.image as img_ops
from wisp.offline_renderer import OfflineRenderer
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from configuration import *


def load_dataset() -> MultiviewDataset:
    transform = SampleRays(num_samples=RAYS_SAMPLED_PER_IMAGE)
    dataset = wisp.datasets.load_multiview_dataset(dataset_path=DATASET_PATH,
                                                   split='train',
                                                   mip=MIP,
                                                   bg_color=BG_COLOUR,
                                                   dataset_num_workers=DATASET_WORKERS,
                                                   transform=transform)

    return dataset


def some_stuff(pipeline, num_angles=180, camera_distance=4):
    dataset = load_dataset()
    renderer = OfflineRenderer(render_res=[1024, 1024], render_batch=4000)
    width, height = 1024,1024

    num_lods = 16
    camera_origin = [-3, 0.65, -3]
    angles = np.pi * 0.1 * np.array(list(range(num_angles + 1)))
    x = -camera_distance * np.sin(angles)
    y = camera_origin[1]
    z = -camera_distance * np.cos(angles)
    for d in range( ):
        out_rgb = []
        for idx in tqdm(range(num_angles + 1), desc=f"Generating 360 Degree of View for LOD {d}"):
            # log_metric_to_wandb(f"LOD-{d}-360-Degree-Scene/step", idx, step=idx)
            out = renderer.shade_images(
                pipeline,
                f=[x[idx], y, z[idx]],
                t=[0,0,0],
                fov=30,
                lod_idx=d
            )
            out = out.image().byte().numpy_dict()
            if out.get('rgb') is not None:
                # log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGB", out['rgb'].T, idx)
                out_rgb.append(Image.fromarray(np.moveaxis(out['rgb'].T, 0, -1)))
            if out.get('rgba') is not None:
                # log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGBA", out['rgba'].T, idx)
                pass
            if out.get('depth') is not None:
                # log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Depth", out['depth'].T, idx)
                pass
            if out.get('normal') is not None:
                # log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Normal", out['normal'].T, idx)
                pass
            if out.get('alpha') is not None:
                # log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Alpha", out['alpha'].T, idx)
                pass
        # wandb.log({})

        rgb_gif = out_rgb[0]
        gif_path = os.path.join(FBUFFERDUMP_PATH, "rgb.gif")
        rgb_gif.save(gif_path, save_all=True, append_images=out_rgb[1:], optimize=False, loop=0)
        # wandb.log({f"360-Degree-Scene/RGB-Rendering/LOD-{d}": wandb.Video(gif_path)})


def load_tracer() -> BaseTracer:
    """ Wisp "Tracers" are responsible for taking input rays, marching them through the neural field to render
    an output RenderBuffer.
    Wisp's implementation of NeRF uses the PackedRFTracer to trace the neural field:
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is employed within the training loop, and is responsible for making use of the neural field's
    grid to generate samples and decode them to pixel values.
    """
    tracer = PackedRFTracer(
        raymarch_type='ray',  # Chooses the ray-marching algorithm
        num_steps=1024,  # Number of steps depends on raymarch_type
        bg_color='white'
    )
    return tracer


def load_pipeline(path, device):
    # should load to gpu by default
    # https://pytorch.org/docs/stable/generated/torch.load.html
    pipeline = torch.load(path)
    # scene_state = WispState()  # Joint trainer / app state
    pipeline.eval()
    return pipeline


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = "20230307-011538"
    path = os.path.join("D:\\", "Workspace", "akw", "kaolin-wisp", "_results",
                        "logs", "runs", "test-nerf", run_name, "model.pth", )

    pipeline = load_pipeline(path=path, device=device)
    # pipeline.to(device)
    count = 4096
    pos = [0.8636, 0.7595, -0.4135]
    poss = [pos * count]
    to = torch.reshape(torch.tensor(poss), (count, -1))
    td = to.clone()
    some_stuff(pipeline=pipeline)
    print("Done Some stuff\n\n\n\n")

    '''
    rays = Rays(origins=to, dirs=td, dist_min=0.0, dist_max=6.0, )
    rays = rays.to(device)
    channels = {"rgb"}
    extra_channels = {}
    rbf = pipeline.forward(rays=rays, channels=channels, extra_channels=extra_channels)
    rbf = rbf.cpu().detach()
    img_ops.write_exr("out.exr", rbf.exr_dict())
    '''
    # rgb = rbf.rgb
    # print("rgb type = ", type(rgb))
    # rgb.cpu().detach().numpy()
    # plt.imshow(rgb)
    # plt.imshow(rgb.permute(1, 2, 0))
    # rbf = rbf.to("cpu")
    # rbf.to("cpu")
    # rbf = rbf.detach().numpy()
    # img_ops.write_exr("out.exr", rbf.exr_dict())

    '''
    # tracer = load_tracer()
    print("\n\n\nAAA\n\n\n")
    scene_state = WispState()
    scene_state.renderer.device = device
    app = WispApp(wisp_state=scene_state)
    app.add_pipeline(pipeline=pipeline, name="Flynn")
    # app.dump_framebuffer(path=FBUFFERDUMP_PATH)
    app.run()
    '''


if __name__ == '__main__':
    print("Load and Show")
    main()
    print("Done")
