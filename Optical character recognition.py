import cv2
import pytesseract
from Levenshtein import distance

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread("Sample Quotes\(0).png")


# get grayscale image
def get_grayscale(photo):
    return cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)


# noise rem oval
def remove_noise(photo):
    return cv2.medianBlur(photo, 5)


# canny edge detection
def canny(photo):
    return cv2.Canny(photo, 100, 200)


# blur
def blur(photo):
    img_blur = cv2.GaussianBlur(photo, (5, 5), 0)
    cv2.imwrite(r"./preprocess/img_blur.png", photo)
    return img_blur


blurred = blur(image)
greyed = get_grayscale(image)
removed_noise = remove_noise(image)
canned = canny(image)

variations = {"Original": image, "Blur": blurred, "Grey": greyed, "Noiseless": removed_noise, "Canny": canned}
prediction_accuracy = {}

img = cv2.resize(image, (1024, 728))
cv2.imshow("Image", img)  # showing resized image

ground = 'Tesseract Sample'

for i in variations:
    text = pytesseract.image_to_string(variations[i])
    print(i + ': \nText prediction: \n' + text + '\nAccuracy: ')
    print(distance(ground, text))  # Calculating Levenshtein distance

    print('\n')

cv2.waitKey(0)
cv2.destroyAllWindows()