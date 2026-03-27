import cv2
import numpy as np
import chess
import chess.pgn

from visualizer import board_to_image


class MoveDetector:
    def __init__(self, cells=8):
        self.board = chess.Board()
        self.prev_img = None
        self.curr_img = None
        self.ref_img = None

        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "CV Chess Game"
        self.game.headers["White"] = "Player (Camera)"
        self.game.headers["Black"] = "Player (Camera)"
        self.node = self.game

        self.cells = cells
        self.grid_h = 500
        self.grid_w = 500
        self.h_grid = np.linspace(0, self.grid_h, self.cells + 1)
        self.v_grid = np.linspace(0, self.grid_w, self.cells + 1)

        self.last_diff_image = None
        self.last_status = "Idle"

    # =========================
    # GRID
    # =========================
    def calibrate_grid(self, warped_board):
        """Default grid split 8x8 by image size."""
        h, w = warped_board.shape[:2]
        self.grid_h = h
        self.grid_w = w
        self.h_grid = np.linspace(0, h, self.cells + 1)
        self.v_grid = np.linspace(0, w, self.cells + 1)
        print(f"Grid calibrated: {w}x{h}, {self.cells}x{self.cells}")

    def _cluster_lines(self, data, max_val):
        if not data:
            return np.linspace(0, max_val, self.cells + 1)
        data.sort()
        res = []
        cluster = [data[0]]
        for i in range(1, len(data)):
            if abs(data[i] - data[i - 1]) > 40:
                res.append(sum(cluster) / len(cluster))
                cluster = [data[i]]
            else:
                cluster.append(data[i])
        res.append(sum(cluster) / len(cluster))

        if len(res) != self.cells + 1:
            return np.linspace(0, max_val, self.cells + 1)
        return np.array(res, dtype=np.float32)

    def calibrate_grid_from_hough(self, warped_board):
        """Grid calibration based on Hough lines (logic from test_chess_thaihung2.py)."""
        h, w = warped_board.shape[:2]
        self.grid_h = h
        self.grid_w = w

        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)

        h_lines, v_lines = [], []
        if lines is not None:
            for l in lines:
                rho, theta = l[0]
                if np.pi / 4 < theta < 3 * np.pi / 4:
                    h_lines.append(abs(rho))
                else:
                    v_lines.append(abs(rho))

        self.h_grid = self._cluster_lines(h_lines, h)
        self.v_grid = self._cluster_lines(v_lines, w)
        print(f"Grid (Hough) calibrated: H={len(self.h_grid)} V={len(self.v_grid)}")

    # =========================
    # GAME STATE
    # =========================
    def _rebuild_game(self):
        """Rebuild PGN tree from current move stack (used after undo/reset)."""
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "CV Chess Game"
        self.game.headers["White"] = "Player (Camera)"
        self.game.headers["Black"] = "Player (Camera)"

        node = self.game
        for mv in self.board.move_stack:
            node = node.add_variation(mv)
        self.node = node

    def get_pgn_string(self):
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return self.game.accept(exporter)

    # =========================
    # FRAME CONTROL
    # =========================
    def set_reference_frame(self, img):
        self.calibrate_grid_from_hough(img)
        self.ref_img = img.copy()
        self.prev_img = img.copy()
        self.curr_img = img.copy()
        self.last_status = "Reference set"
        print("Reference frame set")

    def reset(self):
        self.board = chess.Board()
        self.prev_img = None
        self.curr_img = None
        self.ref_img = None
        self.last_diff_image = None
        self.last_status = "Reset"
        self._rebuild_game()
        print("Game reset")

    def update_frame(self, img):
        if self.prev_img is None:
            self.calibrate_grid(img)
            self.prev_img = img.copy()

        if img.shape[:2] != (self.grid_h, self.grid_w):
            self.calibrate_grid(img)

        self.curr_img = img.copy()

    # =========================
    # DETECTION CORE
    # =========================
    def _prepare_diff(self, prev_img, curr_img):
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        return diff, thresh

    def detect_changes(self, prev_img, curr_img):
        """Return changed squares sorted by intensity (logic from test_chess_thaihung2.py)."""
        if prev_img is None or curr_img is None:
            return [], {}

        diff, thresh = self._prepare_diff(prev_img, curr_img)
        self.last_diff_image = thresh.copy()

        changes = []
        rows = min(self.cells, len(self.h_grid) - 1)
        cols = min(self.cells, len(self.v_grid) - 1)

        for r in range(rows):
            for c in range(cols):
                y0, y1 = int(self.h_grid[r]), int(self.h_grid[r + 1])
                x0, x1 = int(self.v_grid[c]), int(self.v_grid[c + 1])
                y1 = min(y1, thresh.shape[0])
                x1 = min(x1, thresh.shape[1])

                roi = thresh[y0:y1, x0:x1]
                non_zero = cv2.countNonZero(roi)
                print(non_zero)

                if non_zero > 15:
                    sq = chess.square(c, 7 - r)
                    changes.append((sq, non_zero))

        changes.sort(key=lambda x: x[1], reverse=True)
        changed_squares = [sq for sq, _ in changes]
        score_map = {sq: score for sq, score in changes}
        return changed_squares, score_map

    def _expected_squares_for_move(self, move):
        expected = {move.from_square, move.to_square}

        # Castling: rook also moves
        if self.board.is_castling(move):
            if move.to_square == chess.G1:
                expected.add(chess.H1)
                expected.add(chess.F1)
            elif move.to_square == chess.C1:
                expected.add(chess.A1)
                expected.add(chess.D1)
            elif move.to_square == chess.G8:
                expected.add(chess.H8)
                expected.add(chess.F8)
            elif move.to_square == chess.C8:
                expected.add(chess.A8)
                expected.add(chess.D8)

        # En passant: captured pawn is not on destination square before move
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            expected.add(cap_sq)

        return expected

    def infer_move(self, changed_squares, square_scores):
        if len(changed_squares) < 2:
            return None, "Not enough changes"

        top_changes = changed_squares[:4]
        legal_moves = list(self.board.legal_moves)
        possible_moves = []

        for mv in legal_moves:
            if mv.from_square in top_changes and mv.to_square in top_changes:
                possible_moves.append(mv)

        if len(possible_moves) == 1:
            return possible_moves[0], "Success"
        if len(possible_moves) > 1:
            return possible_moves[0], "Ambiguous, picked best match"
        return None, "No matching legal move"

    # =========================
    # ACTIONS
    # =========================
    def confirm_move(self):
        if self.prev_img is None or self.curr_img is None:
            self.last_status = "Need initialization"
            print("Need more frames")
            return None

        changes, scores = self.detect_changes(self.prev_img, self.curr_img)
        move, status = self.infer_move(changes, scores)

        top = [chess.square_name(sq) for sq in changes[:6]]
        print(f"Changes: {top} | Status: {status}")

        if move is None:
            self.last_status = f"Detect fail: {status}"
            print("Detect fail")
            return None

        san = self.board.san(move)
        self.board.push(move)
        self.node = self.node.add_variation(move)
        self.prev_img = self.curr_img.copy()
        self.last_status = f"Move: {san}"

        print(f"Move accepted: {move.uci()} ({san})")
        print(self.get_pgn_string())
        return move

    def undo(self):
        if len(self.board.move_stack) > 0:
            self.board.pop()
            self._rebuild_game()
            self.last_status = "Undo"
            print("Undo")

    # =========================
    # VISUALS
    # =========================
    def get_visual_board(self):
        return board_to_image(self.board, size=500)

    def get_diff_image(self):
        if self.last_diff_image is None:
            return np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)

        diff_bgr = cv2.cvtColor(self.last_diff_image, cv2.COLOR_GRAY2BGR)
        diff_bgr[:, :, 2] = np.maximum(diff_bgr[:, :, 2], self.last_diff_image)
        return diff_bgr

    def draw_grid(self, img):
        img_copy = img.copy()

        if img.shape[:2] != (self.grid_h, self.grid_w):
            self.calibrate_grid(img)

        for x in self.v_grid:
            cv2.line(img_copy, (int(x), 0), (int(x), img_copy.shape[0]), (255, 0, 0), 1)

        for y in self.h_grid:
            cv2.line(img_copy, (0, int(y)), (img_copy.shape[1], int(y)), (0, 255, 0), 1)

        cv2.putText(
            img_copy,
            self.last_status,
            (8, max(22, int(0.045 * img_copy.shape[0]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img_copy