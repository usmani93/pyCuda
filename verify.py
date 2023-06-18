import time
import model as m
import torch as t
import numpy as np
import torchvision as tv
import load_single_image as lsi
import matplotlib.pyplot as plt
from PIL import Image

#to show image
def imshow(sample_element, title):
    plt.imshow(sample_element[0].numpy().transpose(1, 2, 0))
    plt.title(title)
    plt.show()

#transformation for images
transform = tv.transforms.Compose(
    [tv.transforms.ToTensor(),
     tv.transforms.Resize(size=(64,64)),
     tv.transforms.Normalize((0.5,), (0.5,))])

#load model for evaluation
custom_model = m.MPL()
custom_model.load_state_dict(t.load('.\custom'))
print('Model loaded ')
custom_model.eval()

#load image(s) for prediction
single_image = lsi.LoadImages(main_dir='.\Single', transform=transform)
load_single_image = t.utils.data.DataLoader(single_image, shuffle = False)
# image = next(iter(load_single_image))
print('Image(s) loaded ')

#load image into model and get output as prediction
# outputs = custom_model(image)

#load original dataset to get classes
dataset_training = tv.datasets.ImageFolder(root='.\Pictures', transform=transform)
classes = dataset_training.classes
print('Classes loaded ')

def predict_single_image():
    time_prediction_start = time.time()
    image_one = next(iter(load_single_image))
    outputs = custom_model(image_one)
    probs = t.nn.functional.softmax(outputs, dim=1)
    _, predicted = t.max(probs, dim=1)
    time_prediction_end = time.time()
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(1)))
    time_elapsed = time_prediction_end - time_prediction_start
    conf = ' with confidence {0:.2f}'.format(_.item())
    imshow(image_one, f'{classes[predicted.item()]}{conf} in {time_elapsed}')
    img = tv.io.read_image("./Single/79.jpg")
    transform = tv.transforms.Compose([tv.transforms.Resize(size=(64,64))])
    pimg = transform(img)
    ou = custom_model(pimg)
    plt.imshow(np.squeeze(ou.render()))
    plt.show

def predict_list_of_images():
    time_prediction_start = time.time()
    count = 0
    for image in load_single_image:
        print(single_image.total_images[count])
        count += 1
        outputs = custom_model(image)
        probs = t.nn.functional.softmax(outputs, dim=1)
        _, predicted = t.max(probs, 1)
        print(predicted)
        #range(1) for two classes
        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(1)))
        print(f'Confidence: {_.item()}')
        imshow(image, f'{classes[predicted.item()]} {_.item()}')
    time_prediction_end = time.time()
    total_time = time_prediction_end - time_prediction_start
    print('Number of images: {}, time taken: {}'.format(len(load_single_image), total_time))

#predict_list_of_images()
predict_single_image()