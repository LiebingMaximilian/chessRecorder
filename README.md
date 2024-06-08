A python program to track a chessgame using a webcam placed over the center of the board.

Setup:
1. clone
2. install stockfish and enter the path in the config.txt
3. if you do not want to use a camera, but instead use a directory with images, enter the directory path in the config.txt file otherwise leave it as it is.
4. run main.py and hit space whenever you want to record a move. Press f to end and b to take a move back
5. if something is not working you can try to get more information by enabling debug mode in the config.txt



Explanation of the other config params:

  - aspectRatioMin and aspectRationMax: is the minimal and maximal allowed ration of a detected box. Usually pieces should get detected quite "squarish", so this can potentially sort out false positives
  - confidence: is the minimal confidence the object detection needs to recognize a piece. leave this very low. If a piece is not detected you can try to lower this value
  - neededOverlap: the overlapPercentage a piece needs to have with a square, to be standing on that square. lower this for very small pieces, make it higher for very big pieces. But in generall keep it very low
