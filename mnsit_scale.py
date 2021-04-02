import os
import numpy as np
from PIL import Image

def rescale_batch_images(paths, directory_path):
    image_paths = paths
    realization_number = (directory_path.split('/')[1]).split('_')[0]
    for image in image_paths:
        print("Rescaling " + f"{image}" + " realization " + f"{realization_number}")
        old_im = Image.open(f"{image}")
        old_size_initial = old_im.size
        unform_constant = np.random.uniform(0.3,1)

        old_im = old_im.resize((int(old_size_initial[0] * unform_constant), int(old_size_initial[0] * unform_constant)), Image.ANTIALIAS)
        old_size = old_im.size

        new_size = old_size_initial
        new_im = Image.new("RGB", new_size)
        new_im.paste(old_im, ((new_size[0]-old_size[0])//2,
                            (new_size[1]-old_size[1])//2))

        rescaled_image_filename = image.split("/").pop()
        # print(rescaled_image_filename)
        new_im.save(directory_path + "/" + rescaled_image_filename)

def get_images_path(directory_path):
    print("\nGetting file paths...")
    paths = []
    for filename in os.listdir(directory_path):
        f = os.path.join(directory_path, filename)
        if os.path.isfile(f):
            #print(f)
            pass
        paths.append(str(directory_path + "/" + filename))
    print("Returning file paths...\n")
    return sorted(paths)

IMAGES_DIRECTORY_PATH = 'mnist/train/data'
RESCALED_IMAGES_DIRECTORY_PATH ='rescaled_mnist/'
NUM_OF_REALIZATIONS = 6

if __name__ == "__main__":
    images_path = get_images_path(IMAGES_DIRECTORY_PATH)

    for realization in range(0, NUM_OF_REALIZATIONS):
        rescale_batch_images(images_path, str(RESCALED_IMAGES_DIRECTORY_PATH + f"{realization+1}_realization"))
        print()