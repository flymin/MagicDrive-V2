import os
import sys
import fire
from moviepy.editor import ImageSequenceClip


def create_video_from_images(folder_path, bitrate, cut_head, cut_tail, add_subfix=None):
    # Extract video filename and parameters from the folder path
    folder_name = os.path.basename(folder_path)
    parts = os.path.splitext(folder_name)[0].split('_')

    # Extract frame count and FPS
    try:
        fps = int(parts[-1].replace('fps', ''))
        frame_count = int(parts[-2].replace('f', ''))
    except (ValueError, IndexError):
        print("Unable to extract frame count and FPS from the folder name")
        return

    # Get all image files
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
    images.sort()  # Ensure images are in order

    # Check image count
    if len(images) != frame_count:
        print(
            f"Warning: The number of images found ({len(images)}) does not match the specified frame count ({frame_count})")

    # Create video
    image_paths = [os.path.join(folder_path, img) for img in images]
    image_paths = image_paths[cut_head:]
    if cut_tail > 0:
        image_paths = image_paths[:-cut_tail]
    clip = ImageSequenceClip(image_paths, fps=fps)

    # Save video
    subfix = "_convert" if add_subfix is None else f"_{add_subfix}"
    if bitrate is not None:
        subfix += f"_bit{bitrate}"
    if cut_head != 0 or cut_tail != 0:
        subfix += f"_h{cut_head}t{cut_tail}"
    output_video_path = subfix.join(os.path.splitext(folder_path))
    clip.write_videofile(output_video_path, codec='libx264', bitrate=bitrate)

    print(f"Video saved to {output_video_path}")


def main(
    folder_path,
    bitrate=None,
    cut_head=0,
    cut_tail=0,
    subfix=None
):
    folder_path = os.path.normpath(folder_path)
    assert folder_path.endswith(".mp4")
    create_video_from_images(folder_path, bitrate, cut_head, cut_tail, subfix)


if __name__ == "__main__":
    fire.Fire(main)
