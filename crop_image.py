import math
from typing import Any
import cv2
import numpy as np
from PIL import Image


class ImageCropper:
    def __init__(self, min_region_size=10000, border_ratio=0.1):
        self.min_region_size = min_region_size
        self.border_ratio = border_ratio

    def find_contours(self, img):
        """
        find contours limited by min_region_size
        in the binary image.

        The contours are sorted by area size, from large to small.

        Params:
          img: numpy array
        Return:
          boxes: A numpy array of contours.
          each items in the array is a contour (x, y, w, h)
        """
        cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0] if len(cnts) == 2 else cnts[1]

        boxes = []
        copy_img = img.copy()
        for c in cnt:
            (x, y, w, h) = cv2.boundingRect(c)

            if (
                h * w > self.min_region_size
                and h < copy_img.shape[0]
                and w < copy_img.shape[1]
            ):

                # cv2.rectangle(copy_img, (x, y), (x + w, y + h), (155, 155, 0), 1)
                boxes.append([x, y, w, h])

        np_boxes = np.array(boxes)
        # sort the boxes by area size
        area_size = list(map(lambda box: box[2] * box[3], np_boxes))
        area_size = np.array(area_size)
        area_dec_order = area_size.argsort()[::-1]
        sorted_boxes = np_boxes[area_dec_order]

        return sorted_boxes

    def is_intersected(self, new_box, orignal_box) -> bool:
        [x_a, y_a, w_a, h_a] = new_box
        [x_b, y_b, w_b, h_b] = orignal_box

        if y_a > y_b + h_b:
            return False
        if y_a + h_a < y_b:
            return False
        if x_a > x_b + w_b:
            return False
        if x_a + w_a < x_b:
            return False
        return True

    def merge_boxes(self, box_a, box_b) -> list:
        """
        merge 2 intersected box into one
        """
        [x_a, y_a, w_a, h_a] = box_a
        [x_b, y_b, w_b, h_b] = box_b

        min_x = min(x_a, x_b)
        min_y = min(y_a, y_b)
        max_w = max(w_a, w_b, (x_b + w_b - x_a), (x_a + w_a - x_b))
        max_h = max(h_a, h_b, (y_b + h_b - y_a), (y_a + h_a - y_b))

        return [min_x, min_y, max_w, max_h]

    def _remove_borders(self, box) -> Any:
        """
        remove the borders around the box
        """
        [x, y, w, h] = box
        border = math.floor(min(w, h) * self.border_ratio)
        return [x + border, y + border, w - border, h - border]

    def boxes2regions(self, sorted_boxes) -> dict:
        regions = {}

        for box in sorted_boxes:
            if len(regions) == 0:
                regions[0] = box
            else:
                is_merged = False
                for key, region in regions.items():
                    if self.is_intersected(box, region) == True:
                        new_region = self.merge_boxes(region, box)
                        regions[key] = self._remove_borders(new_region)
                        is_merged = True
                        break
                if is_merged == False:
                    key = len(regions)
                    regions[key] = self._remove_borders(box)

        return regions

    def get_cropped_masks(self, mask, regions) -> dict:
        """
        return cropped masks
        """

        results = {}
        for key, region in regions.items():
            [x, y, w, h] = region
            image = Image.fromarray(mask)
            cropped_image = image.crop((x, y, x + w, y + h))
            cropped_mask = np.array(cropped_image)

            results[key] = cropped_mask
        return results

    def merge_regions_and_masks(self, mask, regions) -> dict:
        """
        put regions and masks in a dict, and return it.
        """

        cropped_image = self.get_cropped_masks(mask, regions)
        results = {}

        for key in regions.keys():
            results[key] = {
                "cropped_region": regions[key],
                "cropped_mask": cropped_image[key],
            }

        return results

    def run(self, np_image):
        # find contours
        sorted_boxes = self.find_contours(np_image)

        # get regions
        regions = self.boxes2regions(sorted_boxes)

        # crop regions
        return self.merge_regions_and_masks(np_image, regions)
