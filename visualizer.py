"""
Chess board visualizer — dùng asset PNG thay vì SVG→cairosvg.
[S2] Loại bỏ dependency cairosvg, dùng ChessVisualizer với overlay alpha.
"""
import os
import chess
import cv2
import numpy as np
from config import CFG

PIECE_MAP = {
    'P': 'wp', 'N': 'wn', 'B': 'wb',
    'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb',
    'r': 'br', 'q': 'bq', 'k': 'bk',
}


class ChessVisualizer:
    """Render bàn cờ từ asset PNG — nhanh hơn SVG→PNG 10-50x."""

    def __init__(self, piece_dir=None, square_size=None):
        self.square = square_size or CFG.square_size
        self.pieces = {}
        self.has_assets = True

        piece_dir = piece_dir or CFG.piece_dir

        if not os.path.exists(piece_dir):
            print(f"⚠️ Không tìm thấy thư mục piece: {piece_dir}")
            self.has_assets = False
            return

        for k, v in PIECE_MAP.items():
            path = os.path.join(piece_dir, f"{v}.png")
            # Thử cả lowercase và uppercase extension
            if not os.path.exists(path):
                path = os.path.join(piece_dir, f"{v}.PNG")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"⚠️ Không đọc được piece: {path}")
                self.has_assets = False
            else:
                self.pieces[k] = cv2.resize(img, (self.square, self.square))

    def overlay(self, bg, png, x, y):
        """Overlay ảnh PNG có alpha lên background."""
        s = self.square
        if png.shape[2] == 4:
            alpha = png[:, :, 3] / 255.0
            for c in range(3):
                bg[y:y + s, x:x + s, c] = (
                    alpha * png[:, :, c] +
                    (1 - alpha) * bg[y:y + s, x:x + s, c]
                )
        else:
            bg[y:y + s, x:x + s] = png

    def draw_board(self, board: chess.Board, last_move=None, flip=False):
        """
        Render bàn cờ thành ảnh OpenCV.

        Args:
            board: chess.Board object
            last_move: chess.Move hoặc None — highlight nước đi gần nhất
            flip: True = đen ở dưới (xoay 180°)
        """
        img = np.zeros((8 * self.square, 8 * self.square, 3), dtype=np.uint8)

        # 1. Vẽ ô bàn cờ
        for r in range(8):
            for c in range(8):
                color = (240, 217, 181) if (r + c) % 2 == 0 else (181, 136, 99)

                # Highlight nước đi vừa rồi
                if last_move:
                    sq_idx = chess.square(c, 7 - r) if not flip else chess.square(7 - c, r)
                    if sq_idx == last_move.from_square or sq_idx == last_move.to_square:
                        color = (100, 200, 100)  # Xanh nhạt

                cv2.rectangle(
                    img,
                    (c * self.square, r * self.square),
                    ((c + 1) * self.square, (r + 1) * self.square),
                    color, -1
                )

        # 2. Vẽ quân cờ
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                if flip:
                    r = chess.square_rank(sq)
                    c = 7 - chess.square_file(sq)
                else:
                    r = 7 - chess.square_rank(sq)
                    c = chess.square_file(sq)

                symbol = piece.symbol()
                if self.has_assets and symbol in self.pieces:
                    self.overlay(img, self.pieces[symbol], c * self.square, r * self.square)
                else:
                    # Fallback: vẽ text
                    text_color = (0, 0, 0) if piece.color == chess.BLACK else (255, 255, 255)
                    cv2.putText(img, symbol,
                                (c * self.square + 15, r * self.square + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # 3. Vẽ label file/rank
        label_color = (120, 120, 120)
        for i in range(8):
            file_label = chr(ord('a') + i) if not flip else chr(ord('h') - i)
            rank_label = str(8 - i) if not flip else str(i + 1)
            cv2.putText(img, file_label,
                        (i * self.square + self.square - 14, 8 * self.square - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
            cv2.putText(img, rank_label,
                        (2, i * self.square + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

        return img