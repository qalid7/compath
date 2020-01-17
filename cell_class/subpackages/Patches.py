import numpy as np
from PIL import Image
import math
import csv


class Patches:
    def __init__(self,
                 patch_h,  # Patch H should be odd
                 patch_w,  # Patch W should  be odd
                 num_examples_per_patch=9  # Square Root of Num of Examples must be odd
                 ):
        assert num_examples_per_patch >= 1, 'Number of Examples per patch should be greater than or equal to 1'
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.img_h = None
        self.img_w = None
        self.img_d = None
        self.num_patches_img = None
        self.num_examples_per_patch = num_examples_per_patch
        self.pad_h = math.ceil(patch_h/2.0)
        self.pad_w = math.ceil(patch_h/2.0)

    @staticmethod
    def read_image(input_file):
        image = np.array(Image.open(input_file))
        return image

    @staticmethod
    def read_csv(input_file):
        csv_data = []
        with open(input_file) as csv_file:
            csv_obj = csv.reader(csv_file, delimiter=',')
            next(csv_obj)
            for row in csv_obj:
                csv_data.append(row)
        return csv_data

    def update_variables(self, image):
        self.img_h = np.size(image, 0)
        self.img_w = np.size(image, 1)
        self.img_d = np.size(image, 2)

    def extract_patches(self, input_image, input_csv):
        if type(input_csv) == str:
            csv_data = self.read_csv(input_csv)
        elif type(input_csv) == np.ndarray:
            csv_data = input_csv
        else:
            raise Exception('Please input correct csv path or csv data')

        if type(input_image) == str:
            image = self.read_image(input_image)
        elif type(input_image) == np.ndarray:
            image = input_image
        else:
            raise Exception('Please input correct image path or numpy array')
        self.update_variables(image)

        patch_h = self.patch_h
        patch_w = self.patch_w

        image = np.lib.pad(image, ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w), (0, 0)), 'symmetric')

        self.num_patches_img = len(csv_data) * self.num_examples_per_patch
        img_patches = np.zeros((self.num_patches_img, patch_h, patch_w, image.shape[2]), dtype=image.dtype)
        labels = []
        cell_id = []
        for i in range(self.num_patches_img):
            labels.append([])
            cell_id.append([])
        self.update_variables(image)

        cell_tot = 1
        iter_tot = 0
        for row in csv_data:
            cell_type = row[0]
            cell_location = np.array([row[2], row[1]], dtype=np.int)
            cell_location[0] = cell_location[0] + self.pad_h - 1  # Python index starts from 0
            cell_location[1] = cell_location[1] + self.pad_w - 1  # Python index starts from 0
            if self.num_examples_per_patch > 1:
                root_num_examples = np.sqrt(self.num_examples_per_patch)
                start_location = -int(root_num_examples/2)
                end_location = int(root_num_examples + start_location)
            else:
                start_location = 0
                end_location = 1

            for h in range(start_location, end_location):
                for w in range(start_location, end_location):
                    start_h = cell_location[0] - h - int((patch_h-1)/2)
                    start_w = cell_location[1] - w - int((patch_w-1)/2)
                    end_h = start_h + patch_h
                    end_w = start_w + patch_w
                    labels[iter_tot] = cell_type
                    cell_id[iter_tot] = cell_tot
                    img_patches[iter_tot, :, :, :] = image[start_h:end_h, start_w:end_w]
                    iter_tot += 1

            cell_tot += 1
        return img_patches, labels, cell_id
