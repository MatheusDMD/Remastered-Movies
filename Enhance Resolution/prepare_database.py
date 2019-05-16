import cv2 
import os

# CONSTANTS
NUMBER_OF_IMAGES = 70000
PATH = './unlabeled2017'
NEW_PATH_X = 'dataset/x'
NEW_PATH_Y = 'dataset/y'
NEW_PATH_Z = 'dataset/z'

os.system(f'mkdir -p {NEW_PATH_X}')
os.system(f'mkdir -p {NEW_PATH_Y}')
os.system(f'mkdir -p {NEW_PATH_Z}')

index = 0
images = os.listdir(PATH)
for image in images:
    if index >= NUMBER_OF_IMAGES:
        break
    index +=1
    complete_image_path = os.path.join(PATH, image)
    # print(complete_image_path)
    img = cv2.imread(complete_image_path)
    x,y,z = img.shape
    try:
        crop_img = img[x//2 - 128:x//2 + 128, y//2 - 128:y//2 + 128]
    except Exception as e:
        continue

    if crop_img.shape != (256,256,3):
        print(crop_img.shape)
        continue
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_gray_reduced = cv2.pyrDown(crop_gray)
    # image_reduced = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    path_x = os.path.join(NEW_PATH_X, f'{index}.jpg')
    path_y = os.path.join(NEW_PATH_Y, f'{index}.jpg')
    path_z = os.path.join(NEW_PATH_Z, f'{index}.jpg')
    print(path_x, path_y, path_z)
    cv2.imwrite(path_x, crop_gray_reduced)
    cv2.imwrite(path_y, crop_gray)
    cv2.imwrite(path_z, crop_img)

