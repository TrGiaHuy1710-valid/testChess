import cv2
import numpy as np
import sys
from board_process_en_new import ChessBoardProcessor
from move_detect import MoveDetector
import time
import os

def draw_board_outline(frame, board_contour):
    """Draw detected board outline on camera frame."""
    if board_contour is not None:
        cv2.drawContours(frame, [board_contour], 0, (0, 255, 0), 3)
    return frame

def save_pgn(game, filename="game.pgn"):
    """Save game to PGN file."""
    try:
        with open(filename, "w") as f:
            f.write(str(game))
        print(f"✅ Game saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save PGN: {e}")

def main():
    # ==========================================
    # INITIALIZE COMPONENTS
    # ==========================================
    # Video source: command line argument, or fallback to camera 0
    path = sys.argv[1] if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video source: {path}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    processor = ChessBoardProcessor()
    detector = MoveDetector()
    
    # Frame rate control
    start_time = time.time()
    frame_count = 0
    last_warped_board = None  # Store last processed board
    
    print("=" * 50)
    print("🎮 CHESS BOARD DETECTION & MOVE TRACKING")
    print("=" * 50)
    print("Controls:")
    print("  'i' : Set reference frame for calibration")
    print("  'SPACE' : Confirm detected move")
    print("  'r' : Undo last move")
    print("  'q' : Quit and save game")
    print("=" * 50)
    print()
    
    while True:
        current_time = time.time()
        frame_count += 1
        
        # ==========================================
        # READ FRAME FROM CAMERA
        # ==========================================
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame from camera")
            break
        

        frame = cv2.flip(frame, -1)  # Mirror image for better user experience
        # ==========================================
        # PROCESS BOARD (DETECT + WARP)
        # ==========================================
        # We need to modify ChessBoardProcessor to return both warped board and contour
        # For now, let's process and store the board
        warped_board = processor.process_frame(frame)
        
        if warped_board is not None:
            last_warped_board = warped_board.copy()
            
            # Update detector with new frame
            detector.update_frame(warped_board)
        
        # ==========================================
        # GET VISUAL OUTPUTS
        # ==========================================
        # 1. Camera with board outline
        camera_display = frame.copy()
        orig_h, orig_w = frame.shape[:2]
        display_w, display_h = 640, 480
        if processor.last_board_contour is not None:
            # Scale contour coordinates to match display size
            scale_x = display_w / orig_w
            scale_y = display_h / orig_h
            scaled_contour = (processor.last_board_contour.astype(np.float64) * [scale_x, scale_y]).astype(np.int32)
            camera_display = cv2.resize(camera_display, (display_w, display_h))
            camera_display = draw_board_outline(camera_display, scaled_contour)
        else:
            camera_display = cv2.resize(camera_display, (display_w, display_h))
        
        # 2. Warped board with grid overlay
        if last_warped_board is not None:
            warped_display = detector.draw_grid(last_warped_board)
        else:
            warped_display = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.putText(warped_display, "No board detected", (100, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 3. Chess visual board
        chess_display = detector.get_visual_board()
        
        # 4. Diff image (change detection)
        diff_display = detector.get_diff_image()
        
        # Add info text to displays
        elapsed = current_time - start_time
        fps = max(1, int(frame_count / max(1, elapsed)))
        cv2.putText(camera_display, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(warped_display, f"Board: {detector.board}", (5, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(chess_display, f"Moves: {len(detector.board.move_stack)}", (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ==========================================
        # DISPLAY WINDOWS
        # ==========================================
        cv2.imshow("Camera", camera_display)
        cv2.imshow("Warped Board + Grid", warped_display)
        cv2.imshow("Chess Visual", chess_display)
        cv2.imshow("Diff Detection", diff_display)
        
        # ==========================================
        # KEYBOARD CONTROLS
        # ==========================================
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n🛑 Quitting...")
            # Save game before exit
            pgn_filename = f"game_{int(current_time)}.pgn"
            save_pgn(detector.game, pgn_filename)
            break
        
        elif key == ord('i'):
            # Calibrate - set reference frame
            if last_warped_board is not None:
                detector.set_reference_frame(last_warped_board)
                print(f"FEN: {detector.board.fen()}")
            else:
                print("⚠️ No board detected yet")
        
        elif key == ord(' '):
            # Confirm move
            if last_warped_board is not None:
                move = detector.confirm_move()
                if move:
                    print(f"FEN: {detector.board.fen()}")
            else:
                print("⚠️ No board detected")
        
        elif key == ord('r'):
            # Undo move
            detector.undo()
            print(f"FEN: {detector.board.fen()}")
    
    # ==========================================
    # CLEANUP
    # ==========================================
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")

if __name__ == "__main__":
    main()