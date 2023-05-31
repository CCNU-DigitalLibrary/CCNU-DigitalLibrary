import random
import math
import numpy as np


class MaskingGenerator_block:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                print("delta==0")
                break
            else:
                mask_count += delta

        print("mask_count:", mask_count)
        print("mask:", mask.shape)
        for i in range(0, self.height):
            for j in range(0, self.width):
                print(mask[i][j], end=" ")
            print()

        return mask





class MaskGenerator_simmim_original:
    def __init__(self, input_size=192, mask_patch_size=16, model_patch_size=16, mask_ratio=0.6):
        self.input_size = input_size


        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)


        # print(mask.shape)
        mask_count = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j]==1:
                    mask_count = mask_count+1
        print("mask count;", mask_count)

        return mask


class MaskGenerator_simmim:
    def __init__(self, input_size=(384, 128), mask_patch_size=16, model_patch_size=16, mask_ratio=0.4):
        self.input_size = input_size

        self.height = self.input_size[0]
        self.width = self.input_size[1]


        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.height % self.mask_patch_size == 0
        assert self.width % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        # self.rand_size = self.input_size // self.mask_patch_size


        self.rand_height = self.height // self.mask_patch_size
        self.rand_width = self.width // self.mask_patch_size

        self.scale = self.mask_patch_size // self.model_patch_size

        # self.token_count = self.rand_size ** 2

        self.token_count = self.rand_width * self.rand_height
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        # print("self.mask_count:", self.mask_count)

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_height, self.rand_width))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)


        # print(mask.shape)
        mask_count = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j]==1:
                    mask_count = mask_count+1
        # print(mask)
        return mask
