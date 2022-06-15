import  cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np
from matplotlib import pyplot as plt
import open3d

def DIBR(texture_orig,depth_orig,K_orig,Rt_orig,K_virt,Rt_virt):
    # K_orig  [3 x 3] = Intrinsic Parameters for the original camera
    # Rt_orig [3 x 4] = Extrinsic Parameters for the original camera
    # K_virt  [3 x 3] = Intrinsic Parameters for the virtual camera
    # Rt_virt [3 x 4] = Extrinsic Parameters for the virtual camera
    
    height, width = texture_orig.shape[:2] # height and width of the image

    # Get inverse projection parameters (o = K_orig[R|t]*M)
    fx = K_orig[0,0]  
    fy = K_orig[1,1]
    u0 = K_orig[0,2]  
    v0 = K_orig[1,2]
    P_o = K_orig @ Rt_orig  # matmul multiplication
    P_v = K_virt @ Rt_virt


    # below iterative approach can be done in simple P^-1(3x3)x m_o xDepth
    Cam_XYZ = np.zeros((height*width,3))
    index = 0
    for v in range(height):
        for u in range(width):
            x = (u-u0)*depth_orig[v,u]/fx
            y = (v-v0)*depth_orig[v,u]/fy
            z = depth_orig[v,u]
            Cam_XYZ[index] = (x, y, z)
            index = index+1
    #print(Cam_XYZ)
    Cam_XYZ = np.transpose(Cam_XYZ)

    #visualize3D(Cam_XYZ,250)

    M = np.concatenate((Cam_XYZ,np.ones((1,height*width))),axis=0)

    # Applying a projection matrix of a virutal camera and convert 3D to 2D

    m_V = P_v @ M
    v_points = np.zeros((height*width,2))
    index = 0
    for v in range(height):
        for u in range(width):
            x = m_V[0,index]
            y = m_V[1,index]
            u_P = x/depth_orig[v,u]
            v_P = y/depth_orig[v,u]
            v_points[index] = (v_P, u_P)
            index = index+1
    v_points = np.round(v_points)
    v_points = v_points.astype(np.int64)

    # Round off the pixels in new virutal image and fill cracks with white,
    # create a binary image for Impainting
    index = 0
    myMap = np.zeros((height,width), dtype = np.uint8) # Keeping track of already seen points, a better way would have been use of HashMap for saving space
    output_image = np.zeros((height,width,3))
    bin_image = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            x_o = v_points[index,0]
            y_o = v_points[index,1]
            if(x_o>=0 and x_o<height and y_o>=0 and y_o<width):
                if (myMap[x_o,y_o]==0):
                    output_image[i,j,0] = texture_orig[x_o,y_o,0]
                    output_image[i,j,1] = texture_orig[x_o,y_o,1]
                    output_image[i,j,2] = texture_orig[x_o,y_o,2]
                    myMap[x_o,y_o] = 1
                else:
                    output_image[i,j,0] = 255
                    output_image[i,j,1] = 255
                    output_image[i,j,2] = 255
                    bin_image[i,j] = 1
            index = index+1

    output_image = output_image.astype(np.uint8)

    # save binary image for virtual image
    cv2.imwrite('Virtual_Bin_Image.png', bin_image)
    return output_image

def get_DepthMap(depth_image,z_near, z_far):
    # Function gives depth image (height, weight,3) where each channel pixel has equal value,
    # znear, and zfar vlalue the function generates the depth map

    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    height, weight = depth_image.shape[:2]         
    z_map = np.zeros((height,weight),dtype = np.double)
    depth_image = depth_image.astype(np.double)

    #     For this task inverse mapping is used for depth
    #                         1
    #     Z = _____________________________________________
    #         ((I(ij)/255)x (1/Znear - 1/Zfar)) + (1/Zfar)

    for i in range(height):
        for j in range(weight):
            t1 = depth_image[i,j]/255
            t2 = (1/z_near) - (1/z_far)
            t3 = t1*t2
            t4 = t3+(1/z_far)
            z_map[i,j] = 1/t4
    return z_map

def visualize3D(Cam_XYZ,Distance_Filter):

#     The function takes points projected in XYZ according to depth
#     information and generates a 3D scatter plot. For faster rendering
#     keep Distance_Filter value close to 250, which will discard any points
#     having Z greated than 250, thus eliminating points which are far, the 
#     plot process becomes much faster

    #Select and remove unecessary points which are far
    #toDelete = Cam_XYZ[2]>Distance_Filter
    
    #Cam_XYZ= Cam_XYZ[Cam_XYZ[2,:]<Distance_Filter]
    #Cam_XYZ =  np.delete(Cam_XYZ,np.where(Cam_XYZ < Distance_Filter)[2], axis=0)
    Cam = np.transpose(Cam_XYZ)
    print(Cam.shape)
    #Cam = Cam[np.where(Cam[:,2]<Distance_Filter)]
    print(Cam.shape)

    #Plot with jet color so blue points will be closest
    
    #zscaled = Cam_XYZ[:]
    #cn = np.ceil(max(zscaled))
    #cm = plt.colormap(np.jet(cn))
    #cm = plt.cm.get_cmap("jet")

    #plt.scatter3(Cam_XYZ[0,:],Cam_XYZ[1,:],Cam_XYZ[2,:],[], cm[np.ceil(zscaled),:],'.')
    #3D plot
    #fig = plt.figure(figsize=(12, 12))
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(Cam_XYZ[0,:], Cam_XYZ[1,:], Cam_XYZ[2,:])
    #plt.show()
    #draw_geometries([cloud])
    points = np.random.rand(10000, 3)
    print(points)
    points = Cam
    #print(Cam_XYZ)
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([point_cloud])