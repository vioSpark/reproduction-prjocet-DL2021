import os
import numpy as np
from PIL import Image, ImageStat
from random import randrange
import random
import albumentations as A

def rescale_batch_images(paths, directory_path):
    image_paths = paths
    realization_number = (directory_path.split('/')[1]).split('_')[0]
    for img in image_paths:
        print("Processing " + f"{img}")

        #open image about to be processed
        current_image = Image.open(f"{img}")

        #per-channel mean
        r_mean, g_mean, b_mean = ImageStat.Stat(current_image).mean
        #print("Red-channel mean = " + f"{r_mean}")
        #print("Green-channel mean = " + f"{g_mean}")
        #print("Blue-channel mean = " + f"{b_mean}")

        #per-channel std dev
        r_stddev, g_stddev, b_stddev = ImageStat.Stat(current_image).stddev
        #print("Red-channel std dev = " + f"{r_stddev}")
        #print("Green-channel std dev = " + f"{g_stddev}")
        #print("Blue-channel std dev = " + f"{b_stddev}")

        #get rgb channels
        #r_channel, g_channel, b_channel = current_image.split()
        red_channel = current_image.getchannel('R')
        green_channel = current_image.getchannel('G')
        blue_channel = current_image.getchannel('B')
        
        red_channel = np.asarray(red_channel)
        red_channel = red_channel - red_channel.mean()
        red_channel = red_channel / red_channel.std()

        green_channel = np.asarray(green_channel)
        green_channel = green_channel - green_channel.mean()
        green_channel = green_channel / green_channel.std()

        blue_channel = np.asarray(blue_channel)
        blue_channel = blue_channel - blue_channel.mean()
        blue_channel = blue_channel / blue_channel.std()

        #print(blue_channel)

        r_channel = Image.fromarray(np.uint8(red_channel))
        g_channel = Image.fromarray(np.uint8(green_channel))
        b_channel = Image.fromarray(np.uint8(blue_channel))

        #recombine back to RGB image
        normalized_image = Image.merge('RGB', (r_channel, g_channel, b_channel))
        
        #add 12 pixel zero padding => 96x96 becomes 120x120
        padded_image_size = (int(normalized_image.size[0] + PAD_SIZE*2), int(normalized_image.size[0] + PAD_SIZE*2))

        padded_image = Image.new("RGB", padded_image_size, color = (0, 0, 0))
        padded_image.paste(normalized_image, ((padded_image_size[0]-normalized_image.size[0])//2,
                            (padded_image_size[1]-normalized_image.size[1])//2))

        #random crop padded image
        x, y = padded_image.size

        final_dimension = STL10_IMAGE_DIMENSIONS #96
        x1 = randrange(0, x - final_dimension)
        y1 = randrange(0, y - final_dimension)
    
        croped_image = padded_image.crop((x1, y1, x1 + final_dimension, y1 + final_dimension))    

        #do horizontal flip and add cutout
        croped_image_numpy = np.asarray(croped_image)

        #create horizontal flip function
        transform_horizontal = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])

        #create cutout function
        transform_cutout = A.Compose([
            A.Cutout(num_holes=1, max_h_size=4, max_w_size=8, fill_value=0, p=1)
        ])

        flipped_image = transform_horizontal(image=croped_image_numpy)['image']
        cutout_image = transform_cutout(image=flipped_image)['image']
        #test = transform_cutout(image=croped_image_numpy)['image']

        final_image = Image.fromarray(cutout_image)
        
        final_image_filename = img.split("/").pop()
        final_image.save(directory_path + "/" + final_image_filename)

def get_images_path(directory_path):
    print("\nGetting file paths for")
    paths = []
    for filename in os.listdir(directory_path):
        f = os.path.join(directory_path, filename)
        if os.path.isfile(f):
            #print(f)
            pass
        paths.append(str(directory_path + "/" + filename))
    print("Returning file paths...\n")
    return sorted(paths)

TRAIN_IMAGES_DIRECTORY_PATH = 'original_stl10/train_images'
TEST_IMAGES_DIRECTORY_PATH = 'original_stl10/test_images'
TRAIN_RESCALED_IMAGES_DIRECTORY_PATH ='processed_stl10/rescaled_train_data/'
TEST_RESCALED_IMAGES_DIRECTORY_PATH ='processed_stl10/rescaled_test_data/'
NUM_OF_REALIZATIONS = 1
PAD_SIZE = 12
STL10_IMAGE_DIMENSIONS = 96

if __name__ == "__main__":
    train_images_path = get_images_path(TRAIN_IMAGES_DIRECTORY_PATH)
    test_images_path = get_images_path(TEST_IMAGES_DIRECTORY_PATH)

    for realization in range(0, NUM_OF_REALIZATIONS):
        #rescale_batch_images(images_path, str(RESCALED_IMAGES_DIRECTORY_PATH + f"{realization+1}_realization"))
        rescale_batch_images(train_images_path, str(TRAIN_RESCALED_IMAGES_DIRECTORY_PATH))
        rescale_batch_images(test_images_path, str(TEST_RESCALED_IMAGES_DIRECTORY_PATH))
        print()