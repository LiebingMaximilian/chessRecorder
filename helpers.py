import cv2
import PIL
import numpy as np
from numpy.linalg import norm
import chess
import chess.pgn
import chess.pgn
def getpixelCoordinatesOfField(x,y):
    box = []
    x1 = int(x *11/8 *800)
    x2 = int(x+1 *11/8 *800)
    y1 = 800 - int(y *11/8 *800)
    y2 = 800 - int(y+1 *11/8 *800)

    box.append((x1,y1))
    box.append((x2,y1))
    box.append((x1,y2))
    box.append((x2,y2))
    return box

def drawlinesonimage(image):
    for i in range(1, 8):
        startpoint = (int(i * 1 / 8 * 800), 800)
        endpoint = (int(i * 1 / 8 * 800), 0)
        color = (255, 0, 0)
        image = cv2.line(image, startpoint, endpoint, color, 2)
        startpoint = (800, int(i * 1 / 8 * 800))
        endpoint = (0, int(i * 1 / 8 * 800))
        color = (255, 0, 0)
        image = cv2.line(image, startpoint, endpoint, color, 2)
    return image


def getBbox(result):
    bbox_raw = result.xyxy[0][0]
    bbox = []
    for bound in bbox_raw:
        bbox.append(int(bound.item()))
    return bbox

def drawRectanglesFromResult(result, image, color = (255, 0, 0)):
    bbox = getBbox(result)
    image = image.copy()
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 5)
    return image


def drawRectangles(image, bbox, color):
    return cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 5)


def showImage(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def cropImage(image, bbox):
    return image [bbox [1] : bbox [3], bbox [0]: bbox [2]]


def average_brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

def calculate_overlap(box1, box2):
    # Unpack the box coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the coordinates of the intersection rectangle
    x_overlap_min = max(x1_min, x2_min)
    y_overlap_min = max(y1_min, y2_min)
    x_overlap_max = min(x1_max, x2_max)
    y_overlap_max = min(y1_max, y2_max)

    # Calculate the width and height of the intersection rectangle
    overlap_width = max(0, x_overlap_max - x_overlap_min)
    overlap_height = max(0, y_overlap_max - y_overlap_min)

    # Calculate the area of the intersection rectangle
    overlap_area = overlap_width * overlap_height

    return overlap_area


def getBoxFromIndices(x,y):
    x1 = x * 1/8 * 800
    x2 = (x+1) * 1/8 * 800
    y1 = y * 1/8 * 800
    y2 = (y+1) * 1/8 * 800

    return x1, y1, x2, y2


def checkFieldForPiece(fieldBox, pieces_boxes):
    for box_piece in pieces_boxes:
        overlap = calculate_overlap(fieldBox, box_piece[0])
        overlap_percentage = overlap / 10000
        if overlap_percentage > 0.35:
            return True
    return False

def compare_chessboards(chessboard1, chessboard2):
    """
    Compare two chessboards and return the indices of the squares that have changed.

    Parameters:
    chessboard1 (list of list of int): The first chessboard.
    chessboard2 (list of list of int): The second chessboard.

    Returns:
    list of tuple: A list of tuples where each tuple represents the indices (row, col)
                   of a square that has changed.
    """
    # Initialize a list to store the indices of the changed squares
    changed_squares = []

    # Iterate over each square in the chessboards
    for i in range(8):
        for j in range(8):
            if chessboard1[i][j] != chessboard2[i][j]:
                changed_squares.append((i, j))

    return changed_squares


def index_to_chess_notation(square):
    row, col = square
    """
    Convert a single square's indices to chess notation.

    Parameters:
    row (int): The row index of the square.
    col (int): The column index of the square.

    Returns:
    str: The chess notation of the square.
    """
    # Map columns to letters
    col_to_file = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # Convert column index to file
    file = col_to_file[col]
    # Convert row index to rank
    rank = 8 - row

    # Combine file and rank to get chess notation
    notation = f"{file}{rank}"

    return notation


def uci_to_pgn(moves):
    """
    Convert a list of UCI moves to PGN.

    Parameters:
    moves (list of str): A list of moves in UCI notation.

    Returns:
    str: The PGN representation of the moves.
    """
    # Create a new chess board
    board = chess.Board()

    # Create a game object
    game = chess.pgn.Game()

    # Set up the board
    node = game

    # Push the moves to the board and add them to the game
    for move in moves:
        uci_move = chess.Move.from_uci(move)
        board.push(uci_move)
        node = node.add_variation(uci_move)

    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_string = game.accept(exporter)

    return pgn_string



def printResultFromDetection(boxes, df, shouldprint):
    for index, row in df.iterrows():
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        confidence = row['confidence']
        cls = row['class']
        class_name = row['name']
        boxes.append(((int(xmin), int(ymin), int(xmax), int(ymax)), confidence))
        if shouldprint:
            print(f"Detected {class_name} with confidence {confidence:.2f} at [{xmin}, {ymin}, {xmax}, {ymax}]")
