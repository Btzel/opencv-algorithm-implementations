import cv2
import numpy as np
import math
import os
import scipy.signal
from matplotlib import pyplot as plt


def display(image=None,title="Image",size = 10):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
def generating_kernel(parameter):
    """ Return a 5x5 generating kernel based on an input parameter.
    
    Note: this function is provided for you, do not change it

    Args:
    parameter (float): Range of value [0,1].

    Returns:
    numpy.ndarray: A 5x5 kernel
    """

    kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                       0.25, 0.25 - parameter / 2.0])
    return np.outer(kernel,kernel)


def reduce_img(image):
    """
    Convolve the input image with a generating kernel of parameter of 0.4
    and then reduce its width and height by two

    You can use any / all functions to convolve and reduce the image, although
    the lectures have recommended methods that we advise since there are a lot
    of pieces to this assignment that need to work 'just right'.

    Args:
    image(np.ndarray): a grayscale image of shape (r, c)
    
    Returns:
    output (numpy.ndarray): an image of shape (ceil(r/2), ceil(c/2))
    For instance. if the input is 5x7, the output will be 3x4.
    """

    #per instructionsi, use 0.4 for the kernel generation
    kernel = generating_kernel(0.4)
    #use convolve2d with the image and kernel sent in 
    output = scipy.signal.convolve2d(image,kernel,'same')
    #return every other line and row
    return output[:output.shape[0]:2,:output.shape[1]:2]


def expand(image):
    """
    Expand the image to double size and then convolve it with a generating
    kernel with a parameter of 0.4

    You should upsample the image, and then convolve it with a generating
    kernel of 0.4

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you dont do this, (and i recommend you to try it out without that)
    you will see that your images darken as you apply the convolution.

    Args:
    image (np.ndarray): a grayscale image of shape (r, c)

    Returns:
    output (np.ndarray): an imaeg of shape (2*r, 2*c)
    """
    #per instructions, use 0.4 for the kernel generation
    kernel = generating_kernel(0.4)
    #make a new array double the size, and assign initial values
    output = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    output[:output.shape[0]:2, :output.shape[1]:2] = image
    #use convolve2d to fill in rest
    #multiply by 4 per instructions to scale back up
    output = scipy.signal.convolve2d(output,kernel,'same') * 4
    return output

def gauss_pyramid(image, levels):
    """
    Construct a pyramid from the image by reducing it by the number of
    levels passed in by the input

    Note: you need to use your reduce function in this function to generate
    the output.

    Args:
    image (np.ndarray): a grayscale image of dimension (r, c) and dtype float
    levels (uint8): A positive integer that specifies the number of reductions
    you should do. So, if levels = 0, you should return a list containing just
    the input image. if levels = 1, you should do one reduction
    len(output) = levels + 1

    Returns: 
    output (list): a list of arrays of dtype np.float. The first element of the
    list (output[0]) is layer 0 of the pyramid (the image itself), (output[1])
    is layer 1 of the pyramid (image reduced once), etc. We have already included the
    original image in the output array for you. The arrays are of type numpy.ndarray.
    """

    output = [image]
    for level in range(levels):
        output.append(reduce_img(output[level]))

    return output

def lapl_pyramid(gauss_pyr):
    """
    Construct a laplacian pyramid from gaussian pyramid, of height levels.

    Note: you must use your expand function in this function to generate the
    output. the Gaussian pyramid that is passed in is the output of your gauss_pyramid
    function

    Args:
    gauss_pyr (list): a gaussian pyramid as return by your gauss_pyramid function
    it is a list of numpy.ndarray items.

    Returns:
    output (list): A laplacian pyramid of the same size as gauss_pyr. This
    pyramid should be represented in the same way as gaussPyr, as a list of arrays.
    Every elementof the list now corresponds to a layer of the laplacian pyramid, containing
    the difference between two layers of the gaussian pyramid.

    output[k] = gauss_pyr[k] - expand(gauss_pyr[k+1])

    note: sometimes the size of the expanded image will be larger than the given layer.
    You should crop the expanded image to match in shape with the given layer.

    For example, if my layer is of size 5x7, reducing and expanding will result in an
    image of size 6x8. In this case, crop the expanded layer to 5x7
    """

    output = []
    #look over the lists, but ignore the last element since its cannot be subtracted
    for image1,image2 in zip(gauss_pyr[:-1], gauss_pyr[1:]):
        #add in the subtracted difference
        #expand and bind the 2nd image based on the dimensions of the first
        output.append(
            image1-expand(image2)[:image1.shape[0],:image1.shape[1]]
        )

    #now add the last item back in
    output.append(gauss_pyr[-1])

    return output

def blend(lapl_pyr_white,lapl_pyr_black,gauss_pyr_mask):
    """
    Blend the two laplacian pyramids by weighting them according to the gaussian mask    
    """

    blended_pyr = []
    for lapl_white,lapl_black,gauss_mask in \
            zip(lapl_pyr_white,lapl_pyr_black,gauss_pyr_mask):
        blended_pyr.append(gauss_mask * lapl_white +
                           (1-gauss_mask) * lapl_black)
        
    return blended_pyr

def collapse(pyramid):
    output = pyramid[-1]
    for image in reversed(pyramid[:-1]):
        output = image + expand(output)[:image.shape[0],:image.shape[1]]
    return output


def run_blend(black_image, white_image, mask):
    """ This function administrates the blending of the two images according to
    mask.

    Assume all images are float dtype, and return a float dtype.
    """

    # Automatically figure out the size
    min_size = min(black_image.shape)
    # at least 16x16 at the highest level.
    depth = int(math.floor(math.log(min_size, 2))) - 4

    gauss_pyr_mask = gauss_pyramid(mask, depth)
    gauss_pyr_black = gauss_pyramid(black_image, depth)
    gauss_pyr_white = gauss_pyramid(white_image, depth)

    lapl_pyr_black = lapl_pyramid(gauss_pyr_black)
    lapl_pyr_white = lapl_pyramid(gauss_pyr_white)

    outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
    outimg = collapse(outpyr)

    # blending sometimes results in slightly out of bound numbers.
    outimg[outimg < 0] = 0
    outimg[outimg > 255] = 255
    outimg = outimg.astype(np.uint8)

    return outimg

def get_images(source_folder):
    file_names = os.listdir(source_folder)
    for photo in file_names:
        black_img = cv2.imread('./external/datasets/random/tilt_shift/original/' + photo)
        white_img = cv2.imread('./external/datasets/random/tilt_shift/blur/' + photo)
        mask_img = cv2.imread('./external/datasets/random/tilt_shift/mask/' + photo)
        mask_img.shape = black_img.shape
        print(white_img.shape)
        if mask_img is None:
            continue
        if white_img is None:
            continue
        assert black_img.shape == white_img.shape, \
            "error original and blur size is not equal"
        
        assert black_img.shape == mask_img.shape, \
            "error original and mask size is not equal"
        
        print(photo)
        yield photo,white_img,black_img,mask_img

def main():
    source_folder = os.path.join(script_dir, '../external/datasets/random/tilt_shift/original/')
    for photo,white_img,black_img,mask_img in get_images(source_folder):
        display(black_img)
        print("applying blending")
        black_img = black_img.astype(float)
        white_img = white_img.astype(float)
        mask_img = mask_img.astype(float) / 255
        out_layers = []
        for channel in range(3):
            out_img = run_blend(black_img[:,:,channel],
                                white_img[:,:,channel],
                                mask_img[:,:,channel])
            out_layers.append(out_img)

        out_img = cv2.merge(out_layers)
        display(out_img)
        print("blending done!")

if __name__ == "__main__":
    main()

#its working with a proper mask...