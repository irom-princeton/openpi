"""
Convert DROID data (SVO2 video format) to LeRobot format for pi0.5 fine-tuning.

Data structure expected:
  <data_dir>/
    <episode_timestamp>/
      trajectory.h5
      recordings/SVO/
        <cam_id>.svo2   (MCAP container with HEVC side-by-side stereo video)

Usage:
  uv run examples/droid/convert_droid_svo2_to_lerobot.py --data_dir <path to data directory> --output_path <path to output directory>
"""

import shutil
import subprocess
from pathlib import Path

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
import numpy as np
from tqdm import tqdm
import tyro

FPS = 15
IMAGE_W, IMAGE_H = 320, 180  # DROID RLDS convention

# Camera type codes (from DROID)
WRIST_CAM_TYPE = 0
EXTERIOR_CAM_TYPE = 1


def decode_svo2_frames(svo2_path: Path) -> np.ndarray:
    """
    Decode all frames from a ZED SVO2 file and return them as (N, H, W, 3) RGB uint8.

    SVO2 files are MCAP containers. Each side_by_side message has an 8-byte header
    (two uint32 size fields) followed by an HEVC Annex B chunk. The decoded frame is
    2560×720 (stereo side-by-side); we crop the left half and scale to IMAGE_W×IMAGE_H.
    """
    chunks = []
    with open(svo2_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if "side_by_side" in channel.topic:
                chunks.append(message.data[8:])  # skip 8-byte per-frame header

    if not chunks:
        raise ValueError(f"No side_by_side frames found in {svo2_path}")

    hevc_data = b"".join(chunks)

    # Crop left stereo half (1280×720) and scale to target resolution in one pass
    vf = f"crop=1280:720:0:0,scale={IMAGE_W}:{IMAGE_H}"
    cmd = [
        "ffmpeg", "-f", "hevc", "-i", "pipe:0",
        "-vf", vf,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1", "-v", "quiet",
    ]
    result = subprocess.run(cmd, input=hevc_data, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {svo2_path}: {result.stderr.decode()[:500]}")

    frame_bytes = IMAGE_W * IMAGE_H * 3
    n_frames = len(result.stdout) // frame_bytes
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(n_frames, IMAGE_H, IMAGE_W, 3)


def load_episode(episode_dir: Path) -> dict:
    """Load all data for one episode."""
    h5_path = episode_dir / "trajectory.h5"
    svo2_dir = episode_dir / "recordings" / "SVO"

    with h5py.File(h5_path, "r") as f:
        language_instruction = str(f.attrs.get("current_task", "do something"))
        joint_positions = f["observation/robot_state/joint_positions"][:].astype(np.float32)
        gripper_position = f["observation/robot_state/gripper_position"][:].astype(np.float32)
        joint_velocity = f["action/joint_velocity"][:].astype(np.float32)
        action_gripper = f["action/gripper_position"][:].astype(np.float32)
        cam_types = {
            cam_id: int(f[f"observation/camera_type/{cam_id}"][0])
            for cam_id in f["observation/camera_type"].keys()
        }

    wrist_ids = [k for k, v in cam_types.items() if v == WRIST_CAM_TYPE]
    exterior_ids = sorted([k for k, v in cam_types.items() if v == EXTERIOR_CAM_TYPE])

    if len(wrist_ids) != 1:
        raise ValueError(f"Expected 1 wrist camera, found {wrist_ids} in {episode_dir}")
    if len(exterior_ids) < 2:
        raise ValueError(f"Expected ≥2 exterior cameras, found {exterior_ids} in {episode_dir}")

    wrist_frames = decode_svo2_frames(svo2_dir / f"{wrist_ids[0]}.svo2")
    ext1_frames = decode_svo2_frames(svo2_dir / f"{exterior_ids[0]}.svo2")
    ext2_frames = decode_svo2_frames(svo2_dir / f"{exterior_ids[1]}.svo2")

    n = len(joint_positions)
    assert len(wrist_frames) == n, f"wrist frame count {len(wrist_frames)} ≠ HDF5 length {n}"
    assert len(ext1_frames) == n, f"ext1 frame count {len(ext1_frames)} ≠ HDF5 length {n}"
    assert len(ext2_frames) == n, f"ext2 frame count {len(ext2_frames)} ≠ HDF5 length {n}"

    return {
        "language_instruction": language_instruction,
        "joint_positions": joint_positions,
        "gripper_position": gripper_position,
        "joint_velocity": joint_velocity,
        "action_gripper": action_gripper,
        "wrist_frames": wrist_frames,
        "ext1_frames": ext1_frames,
        "ext2_frames": ext2_frames,
        "n_steps": n,
    }


def main(data_dir: str, output_path: str, *, push_to_hub: bool = False):
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=output_path.name,
        root=output_path,
        robot_type="panda",
        fps=FPS,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (IMAGE_H, IMAGE_W, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (IMAGE_H, IMAGE_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (IMAGE_H, IMAGE_W, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episode_dirs = sorted(d for d in Path(data_dir).iterdir() if (d / "trajectory.h5").exists())
    print(f"Found {len(episode_dirs)} episodes")

    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        try:
            ep = load_episode(ep_dir)
        except Exception as e:
            print(f"Skipping {ep_dir.name}: {e}")
            continue

        print(f"  {ep_dir.name}: {ep['n_steps']} steps — {ep['language_instruction']}")

        for i in range(ep["n_steps"]):
            dataset.add_frame(
                {
                    "exterior_image_1_left": ep["ext1_frames"][i],
                    "exterior_image_2_left": ep["ext2_frames"][i],
                    "wrist_image_left": ep["wrist_frames"][i],
                    "joint_position": ep["joint_positions"][i],
                    "gripper_position": ep["gripper_position"][i : i + 1],
                    "actions": np.concatenate(
                        [ep["joint_velocity"][i], ep["action_gripper"][i : i + 1]],
                        dtype=np.float32,
                    ),
                    "task": ep["language_instruction"],
                }
            )
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda", "svo2"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
