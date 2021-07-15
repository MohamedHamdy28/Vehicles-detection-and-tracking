# Vehicles-detection-and-tracking
This program uses a pre-trained model to detect vehicles on the road and mark their track

# How to give it the input

The main folder is objectdetection.py inside it you can put the path of the video in line 82

After that run the file

I followed the tasks, but I had some issues on how to implement the tracking, so I made a solution that works but not as efficient as in the video

# Note

The length of the life of the track (in case nothing new was assigned to it) has to increase as the number of cars in the video increase, the number put now is good only for the example video, we can solve this problem by assigning ids for each vehicle that appear in the video and set the length of life equal to that number multiplied by some constant, however, I couldn't implement it.

you can change the number in line 159 in objectdetection.py file
