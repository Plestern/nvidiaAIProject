# Endangered Species Sorter

This project takes an image and identifies it using ImageNet's database. Then, it prints whether a species is not at risk, vulnerable, or endangered.

Screenshot of the code and output of an image of a baboon: https://ibb.co/2Nc4gB0


## The Algorithm

The code works using a neural network to compare an image against many others, and using values in many criterion, the code makes its best guess as to what the image is. Then, it runs the name the image was identified as through 3 lists of varied endangerment statuese, each containing many animals. Finally, using list comprehension, the code prints the endangerment status of the animal.


## Running this project

1. Download my-recognition.py and any images(PNG or JPG) that you wish to run.
   
2. In terminal, change directories to access my-recognition.py
   
3. Input the following command to run the script:
    python3 my-recognition.py [YOUR IMAGE HERE]
   
[View a video explanation here](video link)
