"""Projective Homography and Panorama Solution."""
import random

import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        n_size = match_p_src.shape[1]
        A_mat = np.zeros((2*n_size, 9))

        idx = 0
        for i in range(0,n_size):
            x, y = match_p_src[0][i], match_p_src[1][i] # src image
            u, v = match_p_dst[0][i], match_p_dst[1][i] # dst image

            A_mat[idx] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u] # Matrix multiplication from A.1.
            A_mat[idx+1] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v] # Matrix multiplication from A.1.

            idx = idx + 2

        A_mat = np.asarray(A_mat)
        svd_u, svd_s, svd_v = np.linalg.svd(A_mat)  # Using SVD decomposition on A_mat
        normalized_svd_v = svd_v[-1, :] / svd_v[-1, -1]  # Normalize last row
        H = np.reshape(normalized_svd_v, (3, 3))  # Reshape to 3x3

        return H

    @staticmethod
    def compute_forward_homography_slow(
        homography: np.ndarray,
        src_image: np.ndarray,
        dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        row, col = dst_image_shape[0], dst_image_shape[1]
        dst_image = np.zeros((row, col, 3)) # Initialize dst image

        for i in range(src_image.shape[1]): # Loop over cols, x
            for j in range(src_image.shape[0]): # Loop over rows, y
                dot_prod = np.dot(homography, [i, j, 1]) # Dot product of H matrix and location i,j,1 in source_image
                i_dst, j_dst, _ = (dot_prod / dot_prod[2] + 0.5).astype(int) # Normalize
                if (i_dst >= 0 and i_dst < col) and (j_dst >= 0 and j_dst < row): # Make sure we're inside the boundaries
                    dst_image[j_dst, i_dst] = src_image[j, i]

        return dst_image.astype(np.uint8)

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        row_d, col_d = dst_image_shape[0], dst_image_shape[1] # Shape of destination image
        row_s, col_s = src_image.shape[0], src_image.shape[1] # Shape of source image
        dst_image = np.zeros((row_d, col_d, 3)) # Initialize dst image
        [y, x] = np.meshgrid(np.linspace(0, row_s - 1, row_s), np.linspace(0, col_s - 1, col_s)) # Create meshgrid
        x_reshaped, y_reshaped = np.reshape(x.T, row_s * col_s), np.reshape(y.T, row_s * col_s)
        ones_reshaped = np.ones(row_s * col_s)

        image_mat = np.vstack([x_reshaped, y_reshaped, ones_reshaped])
        dst_idx = np.matmul(homography, image_mat) # Transformation using homography matrix
        dst_idx /= dst_idx[2, :] #Normalize

        dst_x, dst_y = np.round(dst_idx[0, :]), np.round(dst_idx[1, :]) # Round coordinates
        inbound = np.argwhere((dst_x >= 0) & (dst_x < col_d) & (dst_y >= 0) & (dst_y < row_d)) # Find locations in the boundaries of the image

        # Insert the source image's corresponding pixels into the destination image
        dst_image[dst_y[inbound].astype(int), dst_x[inbound].astype(int), :] = src_image[
                                                                               image_mat[1, inbound].astype(int),
                                                                               image_mat[0, inbound].astype(int), :]

        return dst_image.astype(np.uint8)

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """

        # Calculate the location of the mapped points
        N = match_p_src.shape[1]
        src_mat = np.concatenate([match_p_src, np.ones((1, N))], axis=0)
        mapped_src_mat = np.matmul(homography, src_mat)
        mapped_src_mat /= mapped_src_mat[2, :]
        mapped_p = mapped_src_mat[0:2, :]

        # Find inliers according to given max_err
        distances = abs(np.sqrt((mapped_p[0, :] - match_p_dst[0, :]) ** 2 + (mapped_p[1, :] - match_p_dst[1, :]) ** 2)) #TODO: Change to hamming distance
        inliers = mapped_p[:, distances <= max_err]

        # Calculate required metrics
        flag = not np.any(inliers)
        if flag:
            fit_percent = 0
            dist_mse = 10**9
        else:
            fit_percent = inliers.shape[1]/N
            dist_mse = np.square(distances[distances <= max_err]).mean()

        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """

        ones_src, ones_dst = np.ones(match_p_src.shape[1]), np.ones(match_p_dst.shape[1])
        src = np.vstack([match_p_src, ones_src])
        dst = np.vstack([match_p_dst, ones_dst])
        mult = np.matmul(homography, src)
        mult = np.round(mult / mult[2, :]) # Normalize
        err = mult - dst
        err_size = err.shape[1]
        norms = np.zeros((1, err.shape[1])) # Initialize
        for i in range(err_size):
            norms[0, i] = np.array(np.linalg.norm(err[:2, i]))
        vectorize_f = np.vectorize(lambda x: x < max_err) # Vectorize if below max_err
        fit_points = vectorize_f(norms)
        mp_src_meets_model, mp_dst_meets_model = match_p_src[:, fit_points.flatten()], match_p_dst[:, fit_points.flatten()]

        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        w = inliers_percent
        t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        p = 0.99
        # # the minimal probability of points which meets with the model
        d = 0.5
        # # number of points sufficient to compute the model
        n = match_p_src.shape[1]
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** 4))) #+ 1

        best_mse = np.inf
        for i in range(k):
            ind_p = np.random.randint(1, n, 4)
            sample_homography = Solution.compute_homography_naive(match_p_src[:, ind_p], match_p_dst[:, ind_p])
            fit_percent, _ = Solution.test_homography(sample_homography, match_p_src, match_p_dst, t)
            mp_src_meets_model, mp_dst_meets_model = Solution.meet_the_model_points(sample_homography, match_p_src,
                                                                                    match_p_dst, t)
            sample_fit_point = self.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
            _, mse_fit_point = self.test_homography(sample_fit_point, mp_src_meets_model, mp_dst_meets_model, t)

            if (mse_fit_point < best_mse) and (fit_percent > d): # If we lowered the mse and our fitting percent is higher
                    homography = sample_fit_point
                    best_mse = mse_fit_point

        try:
            homography
            return homography
        except:
            print('Couldn\'t find a good H, run again')


        #     if fit_percent > d:
        #         all_inliers_homography = Solution.compute_homography_naive(mp_src_meets_model, mp_dst_meets_model)
        #         err = Solution.test_homography(all_inliers_homography, mp_src_meets_model, mp_dst_meets_model, t)[1]
        #         if err < best_err:
        #             homography = all_inliers_homography
        #             best_err = err
        # return homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        row_d, col_d = dst_image_shape[0], dst_image_shape[1]  # Shape of destination image
        row_s, col_s = src_image.shape[0], src_image.shape[1]  # Shape of source image
        src_back_image = np.zeros((row_d, col_d, 3)) # Initialize

        [y, x] = np.meshgrid(np.linspace(0, row_d - 1, row_d), np.linspace(0, col_d - 1, col_d)) # Create meshgrid of cols and rows for destination image
        x_trans, y_trans = x.T, y.T
        x_reshaped, y_reshaped = x_trans.reshape(1, row_d*col_d), y_trans.reshape(1, row_d*col_d)
        ones_reshaped = np.ones((1, row_d*col_d))

        image_mat = np.vstack([x_reshaped, y_reshaped, ones_reshaped])
        dst_idx = np.matmul(backward_projective_homography, image_mat)  # Transformation using homography matrix
        dst_idx /= dst_idx[2, :]  # Normalize

        src_idx_x, src_idx_y = dst_idx[0, :], dst_idx[1, :]
        inbound = np.argwhere((src_idx_x >= 0) & (src_idx_x < col_s) & (src_idx_y >= 0) & (src_idx_y < row_s))
        x_back, y_back = src_idx_x[inbound], src_idx_y[inbound]

        [x, y] = np.meshgrid(np.linspace(0, col_s - 1, col_s), np.linspace(0, row_s - 1, row_s)) # Meshgrid of source image coordinates
        x_reshaped, y_reshaped = x.reshape(row_s*col_s, 1).astype(int), y.reshape(row_s*col_s, 1).astype(int)

        image_values = src_image[y_reshaped, x_reshaped, :]
        image_points = np.hstack([x.reshape(row_s*col_s, 1), y.reshape(row_s*col_s, 1)])

        grid_data = griddata(image_points, image_values, (x_back, y_back), method = 'cubic')
        # Scipy's interpolation.griddata with an appropriate configuration to compute the bi-cubic
        # interpolation of the projected coordinates

        src_back_image[image_mat[1, inbound].astype(int), image_mat[0, inbound].astype(int), :] = grid_data[:, 0, :, :]
        src_back_image = np.clip(src_back_image, 0, 255).astype(np.uint8) # src image backward warped to the destination coordinates

        return src_back_image

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([0, 0, 1])
        src_edges['upper right corner'] = np.array([src_cols_num-1, 0, 1])
        src_edges['lower left corner'] = np.array([0, src_rows_num-1, 1])
        src_edges['lower right corner'] = np.array([src_cols_num-1, src_rows_num-1, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 0:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num-1:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 0:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num-1:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        trans_mat = np.diag([1, 1, 1])
        trans_mat[0, 2] = - pad_left # Padding zeros from left
        trans_mat[1, 2] = - pad_up # Padding zeros from top

        computed_homography = np.matmul(backward_homography, trans_mat) # Compute the homography
        norm_computed_homography = np.linalg.norm(computed_homography)
        final_homography = computed_homography/norm_computed_homography # Normalize

        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        rows_d, cols_d = dst_image.shape[0], dst_image.shape[1]  # Shape of destination image

        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
        panorama_rows, panorama_cols, pad_struct = self.find_panorama_shape(src_image, dst_image, forward_homography)
        backward_homography = np.linalg.inv(forward_homography)
        backward_homography_norm = backward_homography/backward_homography[-1, -1]

        final_homography = self.add_translation_to_backward_homography(backward_homography_norm, pad_struct[3], pad_struct[0])

        src_backward = self.compute_backward_mapping(final_homography, src_image, (panorama_rows, panorama_cols, 3))
        src_backward[pad_struct[0]:rows_d+pad_struct[0], pad_struct[3]:pad_struct[3]+cols_d:] = dst_image
        panorama = np.clip(src_backward, 0, 255).astype(np.uint8)

        return panorama
