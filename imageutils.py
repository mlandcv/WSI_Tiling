import glob
from pathlib import Path
from PIL import Image
import bioformats
import javabridge
from multiprocessing import (Process, Queue, RawArray)
import numpy as np
import functools
import itertools
import time
from typing import (Callable, Dict, List, Tuple)
from imageio import (imsave, imread)
from skimage.measure import block_reduce

from matplotlib import pyplot as plt


def gen_patches(input_folder: Path, output_folder: Path,
                      num_workers: int, by_folder:bool,
                      inverse_overlap_factor: int,
                      patch_size: int, purple_threshold: int,
                      purple_scale_size: int, image_ext: str,series: int,
                      ip_image_ext: str,
                      type_histopath: bool) -> None:
    """
    Generates all patches for subfolders in the training set.
    Args:
        input_folder: Folder containing the subfolders containing WSI.
        output_folder: Folder to save the patches to.
        num_train_per_class: The desired number of training patches per class.
        num_workers: Number of workers to use for IO.
        patch_size: Size of the patches extracted from the WSI.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
    """
    # Find the subfolders and how much patches should overlap for each.
    if ip_image_ext=="czi":
        image_locs = glob.glob("{}/*.czi".format(input_folder))
    
    elif ip_image_ext=="svs":
        image_locs = get_all_image_paths(
            master_folder=input_folder) if by_folder else get_image_names(
                folder=input_folder)
        
    print(f"\ngetting small crops from {len(image_locs)} "
          f"images in {input_folder} "
          f"with inverse overlap factor {inverse_overlap_factor:.2f} "
          f"outputting in {output_folder}")
    
    start_time = time.time()
    outputted_patches = 0

    print("image_locs",image_locs)
    print("image_locs type", type(image_locs))
    
    if ip_image_ext=="czi":
        javabridge.start_vm(class_path=bioformats.JARS, max_heap_size="2G")

        for image_loc in image_locs:
            Image.MAX_IMAGE_PIXELS = None
            
            with bioformats.ImageReader(image_loc) as reader:
                reader.rdr.setSeries(series)

                image = reader.read(image_locs)
                im = image[:, :, 0]

                #plt.figure(figsize=(8,8))
                #plt.plot(im, im)
                #plt.show()
                # print("image_loc=",image_loc)
                # print("image type before raw array=", type(image))
                # print("image shape before raw array=", image.shape)

                img = RawArray(
                    typecode_or_type=np.ctypeslib.as_ctypes_type(dtype=image.dtype),
                    size_or_initializer=image.size)
                img_np = np.frombuffer(buffer=img,
                                    dtype=image.dtype).reshape(image.shape)
                np.copyto(dst=img_np, src=image)

                # print("after np and stuff img_np",type(img_np),img_np.shape)

            
                # Number of x starting points.
                x_steps = int((image.shape[0] - patch_size) / patch_size *
                            inverse_overlap_factor) + 1
                # Number of y starting points.
                y_steps = int((image.shape[1] - patch_size) / patch_size *
                            inverse_overlap_factor) + 1
                # Step size, same for x and y.
                step_size = int(patch_size / inverse_overlap_factor)

                # Create the queues for passing data back and forth.
                in_queue = Queue()
                out_queue = Queue(maxsize=-1)

                # Create the processes for multiprocessing.
                processes = [
                    Process(target=find_patch_mp,
                            args=(functools.partial(
                                find_patch,
                                output_folder=output_folder,
                                image=img_np,
                                by_folder=by_folder,
                                image_loc=image_loc,
                                purple_threshold=purple_threshold,
                                purple_scale_size=purple_scale_size,
                                image_ext=image_ext,
                                type_histopath=type_histopath,
                                patch_size=patch_size), in_queue, out_queue))
                    for __ in range(num_workers)
                ]
                for p in processes:
                    p.daemon = True
                    p.start()

                # Put the (x, y) coordinates in the input queue.
                for xy in itertools.product(range(0, x_steps * step_size, step_size),
                                            range(0, y_steps * step_size, step_size)):
                    in_queue.put(obj=xy)

                # Store num_workers None values so the processes exit when not enough jobs left.
                for __ in range(num_workers):
                    in_queue.put(obj=None)

                num_patches = sum([out_queue.get() for __ in range(x_steps * y_steps)])

                # Join the processes as they finish.
                for p in processes:
                    p.join(timeout=1)
                    outputted_patches += num_patches

            print(
                f"finished patches from {input_folder} "
                f"with inverse overlap factor {inverse_overlap_factor:.2f} in {time.time() - start_time:.2f} seconds "
                f"outputting in {output_folder} "
                f"for {outputted_patches} patches")


    elif ip_image_ext=="svs":
        for image_loc in image_locs:
            Image.MAX_IMAGE_PIXELS = None
            image = imread(
                uri=(image_loc if by_folder else input_folder.joinpath(image_loc)))
            print("image_loc=",image_loc)
            print("image type before raw array=", type(image))
            print("image shape before raw array=", image.shape)

            # Sources:
            # 1. https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
            # 2. https://stackoverflow.com/questions/33247262/the-corresponding-ctypes-type-of-a-numpy-dtype
            # 3. https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
            img = RawArray(
                typecode_or_type=np.ctypeslib.as_ctypes_type(dtype=image.dtype),
                size_or_initializer=image.size)
            img_np = np.frombuffer(buffer=img,
                                dtype=image.dtype).reshape(image.shape)
            np.copyto(dst=img_np, src=image)
            
            print("after np and stuff img_np",type(img_np),img_np.shape)
            # Number of x starting points.
            x_steps = int((image.shape[0] - patch_size) / patch_size *
                        inverse_overlap_factor) + 1
            # Number of y starting points.
            y_steps = int((image.shape[1] - patch_size) / patch_size *
                        inverse_overlap_factor) + 1
            # Step size, same for x and y.
            step_size = int(patch_size / inverse_overlap_factor)

            # Create the queues for passing data back and forth.
            in_queue = Queue()
            out_queue = Queue(maxsize=-1)

            # Create the processes for multiprocessing.
            processes = [
                Process(target=find_patch_mp,
                        args=(functools.partial(
                            find_patch,
                            output_folder=output_folder,
                            image=img_np,
                            by_folder=by_folder,
                            image_loc=image_loc,
                            purple_threshold=purple_threshold,
                            purple_scale_size=purple_scale_size,
                            image_ext=image_ext,
                            type_histopath=type_histopath,
                            patch_size=patch_size), in_queue, out_queue))
                for __ in range(num_workers)
            ]
            for p in processes:
                p.daemon = True
                p.start()

            # Put the (x, y) coordinates in the input queue.
            for xy in itertools.product(range(0, x_steps * step_size, step_size),
                                        range(0, y_steps * step_size, step_size)):
                in_queue.put(obj=xy)

            # Store num_workers None values so the processes exit when not enough jobs left.
            for __ in range(num_workers):
                in_queue.put(obj=None)

            num_patches = sum([out_queue.get() for __ in range(x_steps * y_steps)])

            # Join the processes as they finish.
            for p in processes:
                p.join(timeout=1)

            if by_folder:
                print(f"{image_loc}: num outputted windows: {num_patches}")
            else:
                outputted_patches += num_patches

        if not by_folder:
            print(
                f"finished patches from {input_folder} "
                f"with inverse overlap factor {inverse_overlap_factor:.2f} in {time.time() - start_time:.2f} seconds "
                f"outputting in {output_folder} "
                f"for {outputted_patches} patches")

def find_patch_mp(func: Callable[[Tuple[int, int]], int], in_queue: Queue,
                  out_queue: Queue) -> None:
    """
    Find the patches from the WSI using multiprocessing.
    Helper function to ensure values are sent to each process
    correctly.
    Args:
        func: Function to call in multiprocessing.
        in_queue: Queue containing input data.
        out_queue: Queue to put output in.
    """
    while True:
        xy = in_queue.get()
        if xy is None:
            break
        out_queue.put(obj=func(xy))

                

def find_patch(xy_start: Tuple[int, int], output_folder: Path,
               image: np.ndarray, by_folder: bool, image_loc: Path,
               patch_size: int, image_ext: str, type_histopath: bool,
               purple_threshold: int, purple_scale_size: int) -> int:
    """
    Find the patches for a WSI.
    Args:
        output_folder: Folder to save the patches to.
        image: WSI to extract patches from.
        xy_start: Starting coordinates of the patch.
        by_folder: Whether to generate the patches by folder or by image.
        image_loc: Location of the image to use for creating output filename.
        patch_size: Size of the patches extracted from the WSI.
        image_ext: Image extension for saving patches.
        type_histopath: Only look for purple histopathology images and filter whitespace.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
    Returns:
        The number 1 if the image was saved successfully and a 0 otherwise.
        Used to determine the number of patches produced per WSI.
    """
    x_start, y_start = xy_start

    patch = image[x_start:x_start + patch_size, y_start:y_start +
                  patch_size, :]
    # Sometimes the images are RGBA instead of RGB. Only keep RGB channels.
    patch = patch[..., [0, 1, 2]]

    if by_folder:
        output_subsubfolder = output_folder.joinpath(
            Path(image_loc.name).with_suffix(""))
        output_subsubfolder = output_subsubfolder.joinpath(
            output_subsubfolder.name)
        output_subsubfolder.mkdir(parents=True, exist_ok=True)
        output_path = output_subsubfolder.joinpath(
            f"{str(x_start).zfill(5)};{str(y_start).zfill(5)}.{image_ext}")
    else:
        output_path = output_folder.joinpath(
            f"{image_loc}_{x_start}_{y_start}.{image_ext}")

    if type_histopath:
        if is_purple(crop=patch,
                     purple_threshold=purple_threshold,
                     purple_scale_size=purple_scale_size):
            imsave(uri=output_path, im=patch)
        else:
            return 0
    else:
        imsave(uri=output_path, im=patch)
    return 1

def is_purple(crop: np.ndarray, purple_threshold: int,
              purple_scale_size: int) -> bool:
    """
    Determines if a given portion of an image is purple.
    Args:
        crop: Portion of the image to check for being purple.
        purple_threshold: Number of purple points for region to be considered purple.
        purple_scale_size: Scalar to use for reducing image to check for purple.
    Returns:
        A boolean representing whether the image is purple or not.
    """
    block_size = (crop.shape[0] // purple_scale_size,
                  crop.shape[1] // purple_scale_size, 1)
    pooled = block_reduce(image=crop, block_size=block_size, func=np.average)

    # Calculate boolean arrays for determining if portion is purple.
    r, g, b = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    cond1 = r > g - 10
    cond2 = b > g - 10
    cond3 = ((r + b) / 2) > g + 20

    # Find the indexes of pooled satisfying all 3 conditions.
    pooled = pooled[cond1 & cond2 & cond3]
    num_purple = pooled.shape[0]

    return num_purple > purple_threshold

def get_all_image_paths(master_folder: Path) -> List[Path]:
    """
    Finds all image paths in subfolders.
    Args:
        master_folder: Root folder containing subfolders.
    Returns:
        A list of the paths to the images found in the folder.
    """
    all_paths = []
    subfolders = get_subfolder_paths(folder=master_folder)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            all_paths += get_image_paths(folder=subfolder)
    else:
        all_paths = get_image_paths(folder=master_folder)
    
    all_paths
    return all_paths

def get_subfolder_paths(folder: Path) -> List[Path]:
    """
    Find the paths of subfolders.
    Args:
        folder: Folder to look for subfolders in.
    Returns:
        A list containing the paths of the subfolders.
    """
    return sorted([
        folder.joinpath(f.name) for f in folder.iterdir()
        if ((folder.joinpath(f.name).is_dir()) and (".DS_Store" not in f.name))
    ],
                  key=str)

def get_image_paths(folder: Path) -> List[Path]:
    """
    Find the full paths of the images in a folder.
    Args:
        folder: Folder containing images (assume folder only contains images).
    Returns:
        A list of the full paths to the images in the folder.
    """
    return sorted([
        folder.joinpath(f.name) for f in folder.iterdir() if ((
            folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name))
    ],
                  key=str)

def get_image_names(folder: Path) -> List[Path]:
    """
    Find the names and paths of all of the images in a folder.
    Args:
        folder: Folder containing images (assume folder only contains images).
    Returns:
        A list of the names with paths of the images in a folder.
    """
    return sorted([
        Path(f.name) for f in folder.iterdir() if ((
            folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name))
    ],
                  key=str)