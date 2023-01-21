
# Detection and evaluation of the elements on street imagery taken from a moving vehicle
A object detection project which detect visual pollution from street like:

● GRAFFITI
● FADED SIGNAGE
● POTHOLES
● GARBAGE
● CONSTRUCTION ROAD
● BROKEN_SIGNAGE
● BAD STREETLIGHT
● BAD BILLBOARD
● SAND ON ROAD
● CLUTTER_SIDEWALK
● UNKEPT_FACADE


##  About Dataset
The dataset folder contains the following files:
- images folder (9966 images)
- train.csv
- test.csv
- sample_submission.csv

The columns provided in dataset are as follows:

| Column name  | Description  |
| :------------: | :------------: |
| image_path  | Represent the image name  |
| class  | Represent the class of object detected  |
| name  | Represent the name of object detected  |
| xmax  | Represent the xmax coordinate of bounding box  |
| xmin  | Represent the xmin coordinate of bounding box  |
| ymax  | Represent the ymax coordinate of bounding box  |
| ymin  | Represent the ymin coordinate of bounding box  |


## Problem Statement

Visual pollution is a relatively recent issue compared to other forms of environmental that obstruct natural and man-made landscapes. 
it could have many types such as garbage, graffiti, faded signage, potholes, construction roads, broken signage, bad streetlights, bad billboards, sand on streets, 
cluttered sidewalk, unkept facade, and more and more. which could lead to distraction,
 eye fatigue, decreases in opinion diversity, and loss of identity. It has also been shown 
 to increase biological stress responses and impair balance. So, it's a big problem that faces a lot of cities.
We have faced many problems throughout the journey of implementing our solution which was to try to detect the visual pollution in the pictures. Firstly, while we were exploring and preprocessing the data, we found out that some images have negative coordinates because there are no negative dimensions in the pictures which were making errors and can't detect the object. also, the data was imbalanced, so it was hard to train the model directly, we found that the garbage pollution type had most of the data, unlike other types. And therefore, the prediction wasn't satisfied which was returning a wrong prediction for the image and small accuracy. We also noticed that some pictures have an overlap in the pounding boxes which was to be better if it had only one bounding box.

## Solution

Our solution is a mobile application that uses street imagery to detect and classify visual 
pollution elements, such as garbage, graffiti, faded signage, potholes, construction sites, 
broken signage, faulty streetlights, poor quality billboards, sand on streets, cluttered sidewalks, and poorly maintained building facades. 
The application is easy to use and efficient. Users can either take a picture or upload one, and the model will then detect and classify the visual pollution 
in the image. The image will be displayed with bounding boxes indicating the different types of pollution. The model has 
been trained to achieve an 80% mean average precision.

To develop the solution, we used a variety of tools and frameworks, including Flutter for the user interface, Python, the PyTorch framework, and YOLOv8 for the model's architecture. The deployment was done using Flutter's cross-platform framework to increase accessibility and usability.

During preprocessing, we found that the data was imbalanced, so we decided to divide the model into two stages: detection and classification. The detection stage, which focused on detecting garbage as it had the most data, used the YOLOv8 model with a mean average precision of 80% on the validation dataset..

The classification stage used a combination of supervised and unsupervised learning methods to classify the other types of pollution. Additionally, we also tried the model in a self-supervised manner, using the PyTorch framework, with an F-score matrix of approximately 76%.

Lastly, we combined the two sections of the model to get the final classification and detection of the pollution type in the image. We participated in a hackathon with this solution and are confident it will achieve great results.

##  Working On Data
 - Display some images from data to understand our data like
 ![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/samples/0081af04e33fda23cb1d2da07b994200.jpg)
 ![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/samples/0a2395009a83ce904b3e707a98d78334.jpg)
 ![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/samples/0a281913e0b0d36b1484348ec6014544.jpg)

 - Show bounding box in images
 ![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/bbox/clutter%20sidewalk.png)
 ![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/bbox/potholes.png)
 ![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/bbox/bad%20billboard.png)

- We observe that some bounding boxes are behind each other, so we merge them like that
![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/merge-bbox/b1.jpeg)
![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/merge-bbox/f1.jpeg)

![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/merge-bbox/b2.jpeg)
![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/merge-bbox/f2.jpeg)

![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/merge-bbox/b3.jpeg)
![](https://github.com/HabibaShera/Smartathon/blob/main/imgs/merge-bbox/f3.jpeg)



## Demo

- [Link](https://user-images.githubusercontent.com/73429994/213875075-81ae24f1-8c04-4d8b-9a71-7cf85b2b7fee.mp4)

https://user-images.githubusercontent.com/73429994/213875075-81ae24f1-8c04-4d8b-9a71-7cf85b2b7fee.mp4


