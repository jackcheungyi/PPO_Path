import numpy as np 
from PIL import Image
import cv2 

# im = Image.open("HKSB_6F.pgm")
# im2 = cv2.imread("HKSB_6F_rotated.png")
# print(im2.shape)
# im_array = np.asarray(im)
# print(im_array.shape)
# print(len(im_array[0]))


# print(np.exp(3))
# # cv2.imshow("test",im2)
# # cv2.waitKey(0)
# prev = (3,2)
# cur = (3,3)
# next = (3,4)

# prev_v = np.asarray(cur) -np.asarray(prev)
# next_v = np.asarray(next) - np.asarray(cur)

# dot = np.dot(prev_v,next_v)
# # print(dot)
# magnitude1 = np.linalg.norm(prev_v)
# magnitude2 = np.linalg.norm(next_v)
# cosine_angle = dot / (magnitude1 * magnitude2)
# # Calculate the angle in radians
# angle_rad = np.arccos(cosine_angle)

# # Convert the angle to degrees
# angle_deg = np.degrees(angle_rad)
# if angle_deg == np.nan:
#     print("nan")
# # Print the angle in radians and degrees
# print("Angle (radians):", angle_rad)
# print("Angle (degrees):", angle_deg)

# def linear_schedule(initial_value, final_value=0.0):
#     if isinstance(initial_value, str):
#         initial_value = float(initial_value)
#         final_value = float(final_value)
#         assert (initial_value > 0.0)

#     def scheduler(progress):
#         return final_value + progress * (initial_value - final_value)

#     return scheduler

# # Example usage
# scheduler = linear_schedule(0.15, 0.025)
# for i in range(2049):
#     progress = i / 2048
#     value = scheduler(progress)
#     print(f"Progress: {progress}, Value: {value}")



points = [(1,2),(3,4),(5,6)]
transposed_points = np.transpose(points)
transposed_points[[0, 1]] = transposed_points[[1, 0]]
print(np.transpose(points))
print(transposed_points)