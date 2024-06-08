import os
import traceback
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
from stockfish import Stockfish
from ultralytics import YOLO
import copy
import chess.pgn
from torch.hub import *
import chess.engine


Logger.logInfo("Starting up")
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Read Config
try:
    config = read_config(os.path.join(__location__, 'config.txt'))
    debug = config['Debug'] == 'True'
    stockfish = config['stockfish']
    confidence = float(config['confidence'])
    aspectRatioMin = float(config['aspectRatioMin'])
    aspectRatioMax = float(config['aspectRatioMax'])
    neededOverlap = float(config['neededOverlap'])
except Exception as e:
    Logger.logError('Error reading config. Reason: '+traceback.format_exc())
    exit()

# initialize

board = chess.Board()
game = chess.pgn.Game()
game.setup(board)
cam = cv2.VideoCapture(0)

ucimoveList = []
evallist = []


chessBoardModel = torch.hub.load(__location__, 'custom', os.path.join(__location__, 'chessboardModel'), source="local")
piecesBoardModel = torch.hub.load(__location__, 'custom', os.path.join(__location__, 'piecesModel'), source="local")
piecesBoardModel.conf = confidence
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
        if not config['imagedir'] == '':
            result = True
            image = cv2.imread(os.path.join(config['imagedir'], rf'{halfmovecount}.png'))
        if result:
            # Inference
            result = chessBoardModel(image)
            if debug:
                logInfo('Chessboard recognized: ' + result)

            bbox = helpers.getBbox(result)

            cropped_image = cropImage(image, bbox)

            resized = cv2.resize(cropped_image, (800, 800))

            result2 = piecesBoardModel(resized)
            resized = helpers.drawlinesonimage(resized)

            if debug:
                result2.show()

            boxes = []
            boxesWhite = []
            boxesBlack = []

            df = result2.pandas().xyxy[0]  # DataFrame with columns: xmin, ymin, xmax, ymax, confidence, class, name
            # Loop through the results and print coordinates and class when true(for debugging)
            printResultFromDetection(boxes, df, debug)

            #get average brightness of the whole board
            avgBrightness = helpers.average_brightness(resized)
            boxes.sort(key=lambda tup: tup[1], reverse=True)
            for box in boxes:
                crop = helpers.cropImage(resized, box[0])
                brightness = helpers.average_brightness(crop)
                aspectRatio = (box[0][2] - box[0][0])/(box[0][3] - box[0][1])
                if aspectRatio < aspectRatioMin or aspectRatio > aspectRatioMax:
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
            if debug:
                #showImage(resized)
                print()
            #save the chess position in a matrix 0 means empty, 1 white piece, -1 blackpiece
            chessPositionMatrix = copy.deepcopy(chessboardMatrix)
            #now we iterate over all the fields on the chessboard and look for an overlapp with one of the boxes
            for i in range(8):
                for t in range(8):
                    box = getBoxFromIndices(i , t)
                    haswhitePiece = checkFieldForPiece(box, boxesWhite, neededOverlap)
                    hasblackPiece = checkFieldForPiece(box, boxesBlack, neededOverlap)
                    if haswhitePiece:
                        chessPositionMatrix[t][i] = 1
                    elif hasblackPiece:
                        chessPositionMatrix[t][i] = -1
                    else:
                        chessPositionMatrix[t][i] = 0

            if debug:
                for row in chessPositionMatrix:
                    print(row)

            changed_squares = compare_chessboards(chessboardMatrix, chessPositionMatrix)
            if len(changed_squares) == 0:
                logInfo("Position has not changed")
                return
            #check if the move was to castle:
            if len(changed_squares) == 4:
                if isWhitesMove:
                    if changed_squares.__contains__((7,7)):
                        move = 'e1g1'
                    else:
                        move = 'e1c1'
                else:
                    if changed_squares.__contains__((0,0)):
                        move = 'e8c8'
                    else:
                        move = 'e8g8'
            #this is the case for en passant
            if len(changed_squares) == 3:
                logInfo("changedsquarecount is 3, en passant detected")
                if not board.has_legal_en_passant():
                    logError("en passant was detected, but en passant is not possible in the current position. Something went wrong")
                    for row in chessPositionMatrix:
                        print(row)
                    return
                changed_squares.sort(key = lambda square: square[0])
                changed_squares = find_diagonal_squares(changed_squares)
            if len(changed_squares) == 1:
                logError("only one square has changed, something went wrong")
                showImage(resized)
                result2.show()
                return
            if not len(changed_squares) == 4:
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
            eval = stockfish_evaluation(board, stockfish, 1)
            if eval.is_mate():
                evallist.append(eval.white())
            else:
                rel = eval.white().score()
                evallist.append(rel/100)
            endTime = time.time()
            logInfo(f"Move processed in {endTime-startTime}")
        else:
            raise Exception("could not read Camera image and no imagedirectory was provided in config")
    except Exception as e:
        Logger.logError(traceback.format_exc())

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
    print(evallist)
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
logInfo("Ready")
keyboard.wait()
