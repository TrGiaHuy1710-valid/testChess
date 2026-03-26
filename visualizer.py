import os
import chess
import chess.svg
import cv2
import numpy as np
import cairosvg

PIECE_MAP = {
    'P': 'wp', 'N': 'wn', 'B': 'wb',
    'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb',
    'r': 'br', 'q': 'bq', 'k': 'bk',
}

def board_to_image(board, size=500):
    svg_data = chess.svg.board(board=board, size=size)

    png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))

    np_arr = np.frombuffer(png_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img

class ChessVisualizer:
    def __init__(self, piece_dir="assets/pieces", square_size=80):
        self.square = square_size
        self.pieces = {}

        for k, v in PIECE_MAP.items():
            path = os.path.join(piece_dir, f"{v}.png")
            self.pieces[k] = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def overlay(self, bg, png, x, y):
        s = self.square
        png = cv2.resize(png, (s, s))
        alpha = png[:, :, 3] / 255.0

        for c in range(3):
            bg[y:y+s, x:x+s, c] = (
                alpha * png[:, :, c] +
                (1 - alpha) * bg[y:y+s, x:x+s, c]
            )

    def show(self, board: chess.Board, move=None):
        img = np.zeros((8*self.square, 8*self.square, 3), dtype=np.uint8)

        # draw board
        for r in range(8):
            for c in range(8):
                color = (240, 217, 181) if (r+c) % 2 == 0 else (181, 136, 99)
                cv2.rectangle(
                    img,
                    (c*self.square, r*self.square),
                    ((c+1)*self.square, (r+1)*self.square),
                    color,
                    -1
                )

        # draw pieces
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                self.overlay(
                    img,
                    self.pieces[piece.symbol()],
                    c*self.square,
                    r*self.square
                )

        # highlight move
        if move:
            for s in [move[:2], move[2:4]]:
                sq = chess.parse_square(s)
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                cv2.rectangle(
                    img,
                    (c*self.square, r*self.square),
                    ((c+1)*self.square, (r+1)*self.square),
                    (0, 255, 255),
                    3
                )

        cv2.imshow("Chess Board", img)
        cv2.waitKey(1)

# if __name__ == '__main__':
#     visualizer = ChessVisualizer(piece_dir="../../assets/pieces")
#     board = chess.Board()
#
#     visualizer.show(board)
#     print("Press any key to quit!")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()