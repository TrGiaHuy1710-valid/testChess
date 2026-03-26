import cv2
import numpy as np
import chess
import chess.pgn
from config import CFG
from visualizer import ChessVisualizer


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

        # [S2] Dùng ChessVisualizer thay cairosvg
        self._visualizer = ChessVisualizer()

        # [P1] Cache visual board — chỉ render lại khi FEN thay đổi
        self._cached_visual = None
        self._cached_fen = None

        # [L5] Flag chống ghi đè grid Hough
        self._hough_calibrated = False

        # [F1] Auto-detect state
        self._stable_count = 0
        self._last_change_level = 0.0

        # [F2] Cached inferred move (để vẽ mũi tên preview)
        self._preview_move = None

        # [F3] Flip board
        self.flip_board = False

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
            if abs(data[i] - data[i - 1]) > CFG.hough_cluster_gap:
                res.append(sum(cluster) / len(cluster))
                cluster = [data[i]]
            else:
                cluster.append(data[i])
        res.append(sum(cluster) / len(cluster))

        if len(res) != self.cells + 1:
            return np.linspace(0, max_val, self.cells + 1)
        return np.array(res, dtype=np.float32)

    def calibrate_grid_from_hough(self, warped_board):
        """Grid calibration based on Hough lines."""
        h, w = warped_board.shape[:2]
        self.grid_h = h
        self.grid_w = w

        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, CFG.hough_threshold)

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
        # [L5] Đánh dấu đã calibrate bằng Hough
        self._hough_calibrated = True
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
        self._stable_count = 0
        self.last_status = "Reference set"
        print("Reference frame set")

    def reset(self):
        self.board = chess.Board()
        self.prev_img = None
        self.curr_img = None
        self.ref_img = None
        self.last_diff_image = None
        self.last_status = "Reset"
        self._hough_calibrated = False
        self._cached_fen = None
        self._stable_count = 0
        self._preview_move = None
        self._rebuild_game()
        print("Game reset")

    def update_frame(self, img):
        if self.prev_img is None:
            self.calibrate_grid(img)
            self.prev_img = img.copy()

        # [L5] Không ghi đè grid Hough khi kích thước thay đổi
        if img.shape[:2] != (self.grid_h, self.grid_w) and not self._hough_calibrated:
            self.calibrate_grid(img)

        self.curr_img = img.copy()

    # =========================
    # DETECTION CORE
    # =========================
    def _prepare_diff(self, prev_img, curr_img):
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(prev_gray, CFG.blur_kernel, 0)
        curr_blur = cv2.GaussianBlur(curr_gray, CFG.blur_kernel, 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        _, thresh = cv2.threshold(diff, CFG.diff_threshold, 255, cv2.THRESH_BINARY)
        return diff, thresh

    def detect_changes(self, prev_img, curr_img):
        """Return changed squares sorted by intensity."""
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

                # [L3] Ngưỡng pixel động — tỉ lệ % diện tích ô
                cell_area = max(1, (y1 - y0) * (x1 - x0))
                if non_zero > cell_area * CFG.change_ratio:
                    # [F3] Flip board mapping
                    if self.flip_board:
                        sq = chess.square(7 - c, r)
                    else:
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

        # En passant
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
            expected.add(cap_sq)

        return expected

    def infer_move(self, changed_squares, square_scores):
        if len(changed_squares) < 2:
            return None, "Not enough changes"

        top_changes = set(changed_squares[:CFG.top_changes_count])
        legal_moves = list(self.board.legal_moves)
        scored_moves = []

        for mv in legal_moves:
            expected = self._expected_squares_for_move(mv)
            matched = expected & top_changes
            if len(matched) >= 2 and mv.from_square in top_changes and mv.to_square in top_changes:
                score = len(matched)
                scored_moves.append((mv, score))

        if not scored_moves:
            return None, "No matching legal move"

        # Sort by match score descending
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        # [L4] Xử lý promotion — ưu tiên Queen
        best_score = scored_moves[0][1]
        best_moves = [mv for mv, sc in scored_moves if sc == best_score]

        if len(best_moves) > 1:
            # Ưu tiên queen promotion
            for mv in best_moves:
                if mv.promotion == chess.QUEEN:
                    return mv, "Success (queen promotion)"

        if len(scored_moves) == 1:
            return scored_moves[0][0], "Success"
        if scored_moves[0][1] > scored_moves[1][1]:
            return scored_moves[0][0], "Success (best score)"
        return scored_moves[0][0], "Ambiguous, picked best match"

    # =========================
    # AUTO-DETECT (F1)
    # =========================
    def check_auto_detect(self):
        """
        Kiểm tra xem có nên tự động confirm move không.
        Return True nếu phát hiện thay đổi ổn định (tay đã rút ra).
        """
        if not CFG.auto_detect_enabled:
            return False
        if self.prev_img is None or self.curr_img is None:
            return False
        if self.board.is_game_over():
            return False

        # So sánh curr với prev: nếu có thay đổi lớn → đang di chuyển
        _, thresh = self._prepare_diff(self.prev_img, self.curr_img)
        total_change = cv2.countNonZero(thresh) / max(1, thresh.size)

        if total_change > CFG.auto_change_threshold:
            # Có thay đổi đáng kể → reset stable counter
            self._stable_count = 0
            self._last_change_level = total_change
        else:
            # Ổn định → tăng counter
            if self._last_change_level > CFG.auto_change_threshold:
                # Vừa chuyển từ "có thay đổi" sang "ổn định"
                self._stable_count += 1
            else:
                self._stable_count = 0

        if self._stable_count >= CFG.auto_stable_frames:
            self._stable_count = 0
            return True

        return False

    # =========================
    # ACTIONS
    # =========================
    def confirm_move(self):
        if self.board.is_game_over():
            result = self.board.result()
            self.last_status = f"Game Over: {result}"
            print(f"⚠️ Game already over: {result}")
            return None

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
            self._preview_move = None
            print("Detect fail")
            return None

        san = self.board.san(move)
        self.board.push(move)
        self.node = self.node.add_variation(move)
        self.prev_img = self.curr_img.copy()
        self.last_status = f"Move: {san}"
        self._preview_move = None
        self._cached_fen = None  # Invalidate visual cache

        print(f"Move accepted: {move.uci()} ({san})")
        print(self.get_pgn_string())

        # Check game over after move
        if self.board.is_game_over():
            result = self.board.result()
            reason = ""
            if self.board.is_checkmate():
                reason = "Checkmate"
            elif self.board.is_stalemate():
                reason = "Stalemate"
            elif self.board.is_insufficient_material():
                reason = "Insufficient material"
            elif self.board.is_fifty_moves():
                reason = "Fifty-move rule"
            elif self.board.is_repetition():
                reason = "Threefold repetition"
            self.last_status = f"Game Over: {reason} ({result})"
            print(f"🏁 GAME OVER — {reason}: {result}")

        return move

    def undo(self):
        if len(self.board.move_stack) > 0:
            self.board.pop()
            self._rebuild_game()
            if self.curr_img is not None:
                self.prev_img = self.curr_img.copy()
            self.last_status = "Undo (press 'i' to re-sync)"
            self._cached_fen = None  # Invalidate visual cache
            self._preview_move = None
            print("Undo — press 'i' after restoring the board to re-sync")

    # =========================
    # VISUALS
    # =========================
    def get_visual_board(self):
        """[P1] Cached visual board — chỉ render khi FEN thay đổi."""
        current_fen = self.board.fen()
        if current_fen != self._cached_fen:
            last_move = self.board.peek() if self.board.move_stack else None
            self._cached_visual = self._visualizer.draw_board(
                self.board, last_move=last_move, flip=self.flip_board
            )
            self._cached_fen = current_fen
        return self._cached_visual

    def get_diff_image(self):
        if self.last_diff_image is None:
            return np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)

        diff_bgr = cv2.cvtColor(self.last_diff_image, cv2.COLOR_GRAY2BGR)
        diff_bgr[:, :, 2] = np.maximum(diff_bgr[:, :, 2], self.last_diff_image)
        return diff_bgr

    def _get_cell_center(self, sq):
        """Tính tọa độ trung tâm ô trên warped board."""
        if self.flip_board:
            c = 7 - chess.square_file(sq)
            r = chess.square_rank(sq)
        else:
            c = chess.square_file(sq)
            r = 7 - chess.square_rank(sq)

        if r < len(self.h_grid) - 1 and c < len(self.v_grid) - 1:
            cy = int((self.h_grid[r] + self.h_grid[r + 1]) / 2)
            cx = int((self.v_grid[c] + self.v_grid[c + 1]) / 2)
            return cx, cy
        return None

    def update_preview(self):
        """[F2] Cập nhật preview move dựa trên diff hiện tại."""
        if self.prev_img is None or self.curr_img is None:
            self._preview_move = None
            return
        if self.board.is_game_over():
            self._preview_move = None
            return

        changes, scores = self.detect_changes(self.prev_img, self.curr_img)
        move, status = self.infer_move(changes, scores)
        self._preview_move = move

    def draw_grid(self, img):
        """Vẽ grid + status + preview arrow lên warped board."""
        img_copy = img.copy()

        if img.shape[:2] != (self.grid_h, self.grid_w):
            self.calibrate_grid(img)

        for x in self.v_grid:
            cv2.line(img_copy, (int(x), 0), (int(x), img_copy.shape[0]), (255, 0, 0), 1)

        for y in self.h_grid:
            cv2.line(img_copy, (0, int(y)), (img_copy.shape[1], int(y)), (0, 255, 0), 1)

        # [F2] Vẽ mũi tên preview nếu có
        if self._preview_move is not None:
            pt_from = self._get_cell_center(self._preview_move.from_square)
            pt_to = self._get_cell_center(self._preview_move.to_square)
            if pt_from and pt_to:
                cv2.arrowedLine(img_copy, pt_from, pt_to, (0, 200, 255), 3, tipLength=0.2)
                # Vẽ tên nước đi
                move_text = chess.square_name(self._preview_move.from_square) + \
                            "→" + chess.square_name(self._preview_move.to_square)
                cv2.putText(img_copy, move_text,
                            (8, img_copy.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        # Status text
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
