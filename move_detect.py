import cv2
import numpy as np
import chess
import chess.pgn

from visualizer import board_to_image


class MoveDetector:
    def __init__(self, grid_size=500, cells=8):
        self.board = chess.Board()
        self.prev_img = None
        self.curr_img = None
        self.ref_img = None  # Reference frame for calibration

        self.game = chess.pgn.Game()
        self.node = self.game

        # grid config
        self.grid_size = grid_size
        self.cells = cells

        self.h_grid = np.linspace(0, grid_size, cells + 1)
        self.v_grid = np.linspace(0, grid_size, cells + 1)
        
        # Store last diff for visualization
        self.last_diff_image = None

    # =========================
    # CALIBRATE GRID (DYNAMIC SIZE)
    # =========================
    def calibrate_grid(self, warped_board):
        """Calculate grid lines based on actual warped board dimensions."""
        h, w = warped_board.shape[:2]
        self.grid_size = max(h, w)  # Use actual board size
        self.h_grid = np.linspace(0, h, self.cells + 1)
        self.v_grid = np.linspace(0, w, self.cells + 1)
        print(f"✅ Grid calibrated: {w}x{h}")

    # =========================
    # SET REFERENCE FRAME
    # =========================
    def set_reference_frame(self, img):
        """Set current frame as reference for move detection (press 'i')."""
        self.ref_img = img.copy()
        self.prev_img = img.copy()
        self.curr_img = img.copy()
        print("📸 Reference frame set")

    # =========================
    # RESET
    # =========================
    def reset(self):
        self.__init__()
        print("🔄 Game reset")

    # =========================
    # UPDATE FRAME
    # =========================
    def update_frame(self, img):
        # Calibrate grid on first frame
        if self.prev_img is None:
            self.calibrate_grid(img)
        
        self.curr_img = img.copy()
        if self.prev_img is None:
            self.prev_img = img.copy()

    # =========================
    # DETECT CHANGES (UPDATED)
    # =========================
    def detect_changes(self, prev_img, curr_img):
        if prev_img is None or curr_img is None:
            return []

        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        # 🔥 blur để giảm noise (giữ nguyên logic bạn)
        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        diff = cv2.absdiff(prev_blur, curr_blur)
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        
        # Store diff for visualization
        self.last_diff_image = thresh.copy()

        changes = []

        rows = min(self.cells, len(self.h_grid) - 1)
        cols = min(self.cells, len(self.v_grid) - 1)

        for r in range(rows):
            for c in range(cols):
                y_start, y_end = int(self.h_grid[r]), int(self.h_grid[r + 1])
                x_start, x_end = int(self.v_grid[c]), int(self.v_grid[c + 1])
                
                # Ensure indices are within bounds
                y_end = min(y_end, curr_img.shape[0])
                x_end = min(x_end, curr_img.shape[1])

                roi = thresh[y_start:y_end, x_start:x_end]
                non_zero = cv2.countNonZero(roi)

                if non_zero > 100:
                    sq = chess.square(c, 7 - r)
                    changes.append((sq, non_zero))

        # sort theo độ thay đổi
        changes.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in changes]

    # =========================
    # INFER MOVE (UPDATED)
    # =========================
    def infer_move(self, changed_squares):
        if len(changed_squares) < 2:
            return None, "Not enough changes"

        top_changes = changed_squares[:4]

        legal_moves = list(self.board.legal_moves)
        possible_moves = []

        for move in legal_moves:
            if move.from_square in top_changes and move.to_square in top_changes:
                possible_moves.append(move)

        if len(possible_moves) == 1:
            return possible_moves[0], "Success"

        elif len(possible_moves) > 1:
            return possible_moves[0], "Ambiguous"

        return None, "No matching move"

    # =========================
    # CONFIRM MOVE (SPACE)
    # =========================
    def confirm_move(self):
        if self.prev_img is None or self.curr_img is None:
            print("⚠️ Chưa đủ frame")
            return None

        changes = self.detect_changes(self.prev_img, self.curr_img)
        move, status = self.infer_move(changes)

        print("Changes:", changes[:4], "| Status:", status)

        if move:
            self.board.push(move)
            self.node = self.node.add_variation(move)
            print("✅ Move:", move)

            # update frame sau khi accept move
            self.prev_img = self.curr_img.copy()
        else:
            print("❌ Detect fail")

        return move

    # =========================
    # UNDO
    # =========================
    def undo(self):
        if len(self.board.move_stack) > 0:
            self.board.pop()
            print("↩️ Undo")

    # =========================
    # VISUAL BOARD
    # =========================
    def get_visual_board(self):
        return board_to_image(self.board)

    # =========================
    # GET DIFF IMAGE (VISUALIZATION)
    # =========================
    def get_diff_image(self):
        """Return change detection heatmap for visualization."""
        if self.last_diff_image is None:
            # Return black image if no diff yet
            return np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Convert grayscale to BGR and colorize high-change areas in red
        diff_bgr = cv2.cvtColor(self.last_diff_image, cv2.COLOR_GRAY2BGR)
        # Enhance red channel where changes are detected
        diff_bgr[:, :, 2] = np.maximum(diff_bgr[:, :, 2], self.last_diff_image // 2)
        return diff_bgr

    # =========================
    # DRAW GRID (FROM YOUR CODE)
    # =========================
    def draw_grid(self, img):
        """Draw 8x8 grid on image based on calibrated grid lines."""
        img_copy = img.copy()
        
        for x in self.v_grid:
            cv2.line(img_copy, (int(x), 0), (int(x), img.shape[0]), (255, 0, 0), 1)

        for y in self.h_grid:
            cv2.line(img_copy, (0, int(y)), (img.shape[1], int(y)), (0, 255, 0), 1)

        return img_copy