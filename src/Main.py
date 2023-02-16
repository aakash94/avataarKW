import torch.nn
import sys
from glob import glob
from pathlib import Path
from configuration import *

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil


def do_system(command):
    print(f"==== running: {command}")
    err = os.system(command)
    if err:
        print("FATAL: command failed ", command)
        sys.exit(err)


def run_ffmpeg():
    ffmpeg_binary = "ffmpeg"

    images = IMAGES_PATH
    video = IN_VIDEO_PATH
    fps = float(FPS_COUNT) or 1.0
    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")

    try:
        # Passing Images' Path Without Double Quotes
        shutil.rmtree(images)
    except:
        pass
    do_system(f"mkdir {images}")

    time_slice_value = ""
    time_slice = ""
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"
    do_system(f"{ffmpeg_binary} -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")


def run_colmap():
    colmap_binary = COLMAPBAT_PATH

    db = COLMAPDB_PATH
    images = IMAGES_PATH
    db_noext = COLMAPDB_DIR_PATH
    colmap_camera_model = "OPENCV"
    colmap_camera_params = ""
    colmap_matcher = "sequential"
    text = db_noext + "colmap_text"
    sparse = db_noext + "_sparse"
    print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")

    try:
        os.remove(db)
    except:
        print("Error removing, but continuing nonetheless.")
    do_system(
        f"{colmap_binary} feature_extractor --ImageReader.camera_model {colmap_camera_model} --ImageReader.camera_params \"{colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
    match_cmd = f"{colmap_binary} {colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"

    do_system(match_cmd)
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")
    do_system(f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {sparse}")
    do_system(
        f"{colmap_binary} bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    do_system(f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob,
                          db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def create_dataset():
    run_ffmpeg()
    run_colmap()
    AABB_SCALE = SCALE_AABB
    SKIP_EARLY = 0
    IMAGE_FOLDER = IMAGES_PATH
    TEXT_FOLDER = TEXT_FOLDER_PATH
    OUT_PATH = JSON_PATH
    keep_colmap_coords = False

    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            k3 = 0
            k4 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            is_fisheye = False
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                is_fisheye = True
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                k3 = float(els[10])
                k4 = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(
        f"camera:\n\tres={w, h}\n\tcenter={cx, cy}\n\tfocal={fl_x, fl_y}\n\tfov={fovx, fovy}\n\tk={k1, k2} p={p1, p2} ")

    with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "p1": p1,
            "p2": p2,
            "is_fisheye": is_fisheye,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY * 2:
                continue
            if i % 2 == 1:
                elems = line.split(
                    " ")  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                # name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                print(name, "sharpness=", b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                if not keep_colmap_coords:
                    c2w[0:3, 2] *= -1  # flip the y and z axis
                    c2w[0:3, 1] *= -1
                    c2w = c2w[[1, 0, 2, 3], :]
                    c2w[2, :] *= -1  # flip whole world upside down

                    up += c2w[0:3, 1]

                frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
                out["frames"].append(frame)
    nframes = len(out["frames"])

    if keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)  # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3, :]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3, :]
                p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p * w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp)  # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)


def main():
    create_dataset()


if __name__ == '__main__':
    print("Hello World")
    main()
