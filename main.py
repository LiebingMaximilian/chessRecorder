import torch
import time
import chess
import PIL
import cv2
import Logger
import keyboard
import helpers
import numpy
from helpers import *
from Logger import *
from chess import *
from ultralytics import YOLO
import copy
import chess.pgn
from torch.hub import *

#import functions
calculate_overlap = helpers.calculate_overlap

cropImage = helpers.cropImage
getBoxFromIndices = helpers.getBoxFromIndices
checkFieldForPiece = helpers.checkFieldForPiece


# initialize
Logger.logInfo("Starting up")
board = chess.Board()
game = chess.pgn.Game()
game.setup(board)
cam = cv2.VideoCapture(0)
ucimoveList = []
chessBoardModel = torch.hub.load(r"C:\Users\miebi\yolov5",'custom', r"C:\Users\miebi\yolov5\runs\train\chessboard\weights\best", source="local") #pain in the ass
piecesBoardModel = torch.hub.load(r"C:\Users\miebi\yolov5",'custom', r"C:\Users\miebi\yolov5\runs\train\pieces\weights\best", source="local")
piecesBoardModel.conf = 0.1
chessboardMatrix = [
    [-1, -1, -1, -1, -1, -1, -1, -1],  # Black pieces
    [-1, -1, -1, -1, -1, -1, -1, -1],  # Black pawns
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
    [1, 1, 1, 1, 1, 1, 1, 1],  # White pawns
    [1, 1, 1, 1, 1, 1, 1, 1]  # White pieces
]
playingWhite = True
isWhitesMove = True
halfmovecount = 0

def recordMove():
    global isWhitesMove
    global chessboardMatrix
    global halfmovecount
    startTime = time.time()
    Logger.logInfo('Starting to record move')
    try:
        # getImage
        result, image = cam.read()
        #for testing just use this
        result = True
        image = cv2.imread(fr"C:\Users\miebi\OneDrive\Bilder\friedLiver\{halfmovecount}.jpg")
        if result:
            # Inference
            result = chessBoardModel(image)
            print(result)
            bbox = helpers.getBbox(result)
            cropped_image = cropImage(image, bbox)

            resized = cv2.resize(cropped_image, (800, 800))

            result2 = piecesBoardModel(resized)
            resized = helpers.drawlinesonimage(resized)
            result2.show()

            boxes = []
            boxesWhite = []
            boxesBlack = []

            df = result2.pandas().xyxy[0]  # DataFrame with columns: xmin, ymin, xmax, ymax, confidence, class, name
            # Loop through the results and print coordinates and class when true(for debugging)
            printResultFromDetection(boxes, df, False)


            #get average brightness of the whole board
            avgBrightness = helpers.average_brightness(resized)
            boxes.sort(key=lambda tup: tup[1], reverse=True)
            for box in boxes:
                crop = helpers.cropImage(resized, box[0])
                brightness = helpers.average_brightness(crop)
                cv2.imwrite(r'C:\Users\miebi\Desktop\cropped.jpg', crop)
                aspectRatio = (box[0][2] - box[0][0])/(box[0][3] - box[0][1])
                if aspectRatio < 0.7 or aspectRatio > 1.5:
                    continue
                if brightness < avgBrightness:
                    boxesBlack.append(box)
                else:
                    boxesWhite.append(box)
            for box in boxesBlack:
                box = box[0]
                resized = helpers.drawRectangles(resized, box, (0,0,0))
            for box in boxesWhite:
                box = box[0]
                resized = helpers.drawRectangles(resized, box,  (255,255,255))
            #show for debug
            #showImage(resized)

            #save the chess position in a matrix 0 means empty, 1 white piece, -1 blackpiece
            chessPositionMatrix = copy.deepcopy(chessboardMatrix)
            #now we iterate over all the fields on the chessboard and look for an overlapp with one of the boxes
            for i in range(8):
                for t in range(8):
                    box = getBoxFromIndices(i,t)
                    haswhitePiece = checkFieldForPiece(box, boxesWhite)
                    hasblackPiece = checkFieldForPiece(box, boxesBlack)
                    if haswhitePiece:
                        chessPositionMatrix[t][i] = 1
                    elif hasblackPiece:
                        chessPositionMatrix[t][i] = -1
                    else:
                        chessPositionMatrix[t][i] = 0
            #for row in chessPositionMatrix:
                #print(row)

            changed_squares = compare_chessboards(chessboardMatrix, chessPositionMatrix)
            #check if the move was to castle:
            if len(changed_squares) == 4:
                if isWhitesMove:
                    if changed_squares.__contains__((7,7)):
                        move = 'e1g1'
                    else:
                        move = 'e1c1'
                else:
                    if changed_squares.__contains__((0,0)):
                        move = 'e1g1'
                    else:
                        move = 'e1c1'

            if len(changed_squares) ==1:
                logError("only one square has changed, something went wrong")
                showImage(resized)
                result2.show()
            colorOfPieceThatmoved = 1
            if not isWhitesMove:
                colorOfPieceThatmoved = -1
            if chessPositionMatrix[changed_squares[0][0]][changed_squares[0][1]] == colorOfPieceThatmoved:
                startingSquare = changed_squares[1]
                endSquare = changed_squares[0]
            else:
                startingSquare = changed_squares[0]
                endSquare = changed_squares[1]

            #map the squares to chess notation
            startingSquare_notation = index_to_chess_notation(startingSquare)
            endSquare_notation = index_to_chess_notation(endSquare)


            # check if that move is legal:
            move = startingSquare_notation + endSquare_notation
            pychessmove = Move.from_uci(move)
            if board.legal_moves.__contains__(pychessmove):
                logInfo(f"recognized the move {move}")
            else:
                logError(f"the recognized move {move} is not legal in the current position")
                return

            #game.add_main_variation(pychessmove)
            board.push(pychessmove)
            print(board)
            ucimoveList.append(move)
            chessboardMatrix = chessPositionMatrix
            # Results
            isWhitesMove = not isWhitesMove
            halfmovecount = halfmovecount + 1
            if board.is_checkmate():
                end()
            endTime = time.time()
            logInfo(f"Move processed in {endTime-startTime}")
        else:
            raise Exception("could not read Camera image")
    except Exception as e:
        Logger.logError(e.__str__())

    return


def end():
    logInfo("Game ended")
    if board. is_checkmate():
        logInfo("Game ended with checkmate")
        if isWhitesMove:
            logInfo("Black wins")
        else:
            logInfo("White wins")
    print(ucimoveList)
    print(uci_to_pgn(ucimoveList))
    exit()

def takeMoveBack():
    global ucimoveList
    logInfo("taking move back")
    ucimoveList.pop()
    board.pop()


keyboard.add_hotkey('space', recordMove)
keyboard.add_hotkey('f', end)
keyboard.add_hotkey('b',takeMoveBack)
keyboard.wait()
