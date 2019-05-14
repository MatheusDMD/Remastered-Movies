from PIL import Image
PATH = './dataset/z_small/'
NUMBER_OF_IMAGES = 100000


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()

def crop_bw():
    image_obj = Image.open(image_path).convert('LA')
    width, height = image_obj.size
    if width > 256 or height > 256:
        cropped_image = image_obj.crop(coords)
        cropped_image.save(saved_location)
        cropped_image.show()
    
if __name__ == '__main__':
    index = 0
    images = os.listdir(PATH)
    for image in images:
        if index >= NUMBER_OF_IMAGES:
            break
        index +=1
        complete_image_path = os.path.join(PATH, image)
        crop(complete_image_path, (0, 0, 256, 256), f'{index}.jpg')