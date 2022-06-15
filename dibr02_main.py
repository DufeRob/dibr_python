import cv2
import numpy as np
from matplotlib import pyplot as plt
import dibr02_dibr as dibr

file_depth = 'D_original.png'
file_texture = 'V_original.png'
save_output = True

#     Original Camera Parameters
# K_orig  [3 x 3] = Intrinsic Parameters for the original camera
# Rt_orig [3 x 4] = Extrinsic Parameters for the original camera


K_original = np.array([ [1732.87,   0.0,     943.23],
                        [0.0,       1729.90, 548.845040],
                        [0,         0,       1]])

Rt_original = np.array([[1.0,   0.0, 0.0, 0.0],
                        [0.0,   1.0, 0.0, 0.0],
                        [0.0,   0.0, 1.0, 0.0]])
Zfar = 2760.510889
Znear = 34.506386

#     Virtual Camera Parameters
# K_virt  [3 x 3] = Intrinsic Parameters for the virtual camera
# Rt_virt [3 x 4] = Extrinsic Parameters for the virtual camera

K_virtual = np.array([  [1732.87,    0.0,     943.23],
                        [0.0,        1729.90, 548.845040],
                        [0,          0,       1]])

Rt_virtual = np.array([ [1.0, 0.0, 0.0, 1.5924],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]])

image_original = cv2.imread(file_texture); # read RGB image
image_depth = cv2.imread(file_depth)      # read Depth image
Z_Map = dibr.get_DepthMap(image_depth,Znear, Zfar); # create actual depth map for DIBR function

image_output = dibr.DIBR(image_original,Z_Map,K_original, Rt_original,K_virtual, Rt_virtual); # Invoke DIBR

# If saving output is required then save the image
if save_output:
    cv2.imwrite('Output_Virtual_Image.png',image_output)

#     Plot Input vs Output Image
plt.subplot(1,2,1),plt.imshow(image_original),plt.title('Input Image')
plt.subplot(1,2,2),plt.imshow(image_output),plt.title('Output Image')
plt.show()