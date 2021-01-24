# Prime Control
Level 1 Self Driving

### Data sources:
- https://github.com/commaai/comma2k19
- https://research.comma.ai/
- https://github.com/udacity/self-driving-car/tree/master/datasets

### Data transformations:
1. downsize to (70, 160)
2. RGB to grayscale

### Auxiliary pretraining idea (not currently in use):
The reason why humans can learn to drive with a small amount of driving data is because we utilize information from other parts of our lives. This is why self-driving models that are pre-trained on datasets like ImageNet tend to be more successful on smaller amounts of data. What I wanted to try is to pre-train a model on a task that is maybe a little more related to driving than arbitrary image classification. The idea is to develop 2 stages of pre-training with increasingly complex tasks and data that can be labeled easily

##### Stage 1:
  Predict if the image is displaying the perspective of a driver in a car. The dataset for this task would be dash cam car footage labeled as 1 and other things humans would see labeled as 0. The "other" things would need to have a large amount of variance so I decided to go for compilations of funny videos, animals, and beautiful landscapes. Pretty interesting to see if this would give a boost during training. I believe that it would in fact give a boost because the network would learn patterns that are associated with recognizing lane lines and other things important to driving, and on the other hand it would learn patterns that are associated with human reality.

##### Stage 2:
  Predict if the image is displaying the perspective of a driver in the correct lane or not in the correct lane. The dataset would be regular dash cam footage once again labeled as 1 and flipped images as 0 (as you would be in the opposite lane which is usually not good). The problem here is that some datasets contain images from highways with 4 lanes, and in that case it's much harder to tell if you're in the correct lane or not because you'd only see a sliver of the drivers driving in the opposite direction on your right rather than on your left (due to the width of your 4 lanes). The solution to this would be to filter the dataset or use dash cam footage from countries where people drive on the left side of the road. This task is much more complex than the previous one and it would cause the network to learn to distinguish proper driving from improper driving in the most obvious case and this might help in the final stage.
  
##### Stage 3:
  Predicting the steering angle of a car based on the image data. Using the weights of the pre-trained models, and the aforementioned datasets, learn how to predict the steering angle by minimizing the MSE loss.

### Safe steering angle update rule:
In order to make driving much more safe and smooth I decay the current angle and update it with a fraction of the predicted angle:

current_angle = a * current_angle + (1 - a) * predicted_angle        | where 0.7 < a < 0.9

### Progress:
Model seems to be good at controlling the steering wheel. Currently working on the hardware aspect so I can use the model.
