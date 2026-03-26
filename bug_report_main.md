# 🐛 Bug Report — `main.py` (Luồng Modular)

> Tổng hợp tất cả lỗi đã phát hiện và trạng thái sửa chữa.

---

## Bug 1: `prev_img` Không Được Reset Khi Undo ✅ ĐÃ SỬA

**File**: [move_detect.py — undo()](file:///c:/Users/Admin/BTL_XLA/testChess/move_detect.py#L249-L259)

**Vấn đề**: `undo()` chỉ hoàn tác logic cờ (`board.pop()`) mà không reset `prev_img` → board và ảnh lệch nhau.

**Cách sửa**: Thêm `self.prev_img = self.curr_img.copy()` trong `undo()`, kèm status thông báo user nhấn `'i'` để đồng bộ lại sau khi đặt lại quân trên bàn thật.

```diff
 def undo(self):
     if len(self.board.move_stack) > 0:
         self.board.pop()
         self._rebuild_game()
-        self.last_status = "Undo"
-        print("Undo")
+        if self.curr_img is not None:
+            self.prev_img = self.curr_img.copy()
+        self.last_status = "Undo (press 'i' to re-sync)"
+        print("Undo — press 'i' after restoring the board to re-sync")
```

---

## Bug 2: Logic Nhập Thành / En Passant Chưa Tích Hợp ✅ ĐÃ SỬA

**File**: [move_detect.py — infer_move()](file:///c:/Users/Admin/BTL_XLA/testChess/move_detect.py#L201-L228)

**Vấn đề**: `_expected_squares_for_move()` đã viết nhưng không được gọi → castling/en passant có thể detect sai.

**Cách sửa**: Viết lại `infer_move()`:
- Mở rộng top changes từ 4 → **6 ô** (vì castling thay đổi 4 ô)
- Dùng `_expected_squares_for_move(mv)` để tính tập ô kỳ vọng
- **Scoring**: match count (`len(expected ∩ top_changes)`) → nước đi khớp nhiều ô nhất thắng
- Vẫn yêu cầu `from_square` và `to_square` phải nằm trong top changes

---

## Bug 3: `cv2.waitKey()` Gọi 2 Lần → Mất Phím ✅ ĐÃ SỬA

**File**: [main.py](file:///c:/Users/Admin/BTL_XLA/testChess/main.py#L139)

**Vấn đề**: `waitKey()` gọi 2 lần (dòng 72 + dòng 140), lần 1 tiêu thụ phím → lần 2 ghi đè key = -1.

**Cách sửa**: Xóa `waitKey(delay)` ở đầu loop. Chỉ giữ 1 lần `waitKey(1)` sau `imshow` (dòng 139).

---

## Bug 4: Hardcoded Video Path ✅ ĐÃ SỬA

**File**: [main.py — dòng 29](file:///c:/Users/Admin/BTL_XLA/testChess/main.py#L29)

**Vấn đề**: `path = r"E:\Python_Project\chessboard_move.mp4"` — crash trên máy khác.

**Cách sửa**: `path = sys.argv[1] if len(sys.argv) > 1 else 0` — dùng argument dòng lệnh hoặc fallback camera 0.

---

## Bug 5: `os.remove()` Hardcoded Trong `__main__` ✅ ĐÃ SỬA

**File**: [main.py — dòng 177-178](file:///c:/Users/Admin/BTL_XLA/testChess/main.py#L177)

**Vấn đề**: `os.remove(r"E:\...\inner_pts.npy")` → `FileNotFoundError` crash.

**Cách sửa**: Xóa dòng `os.remove()`. Giờ chỉ gọi `main()`.

---

## Bug 6: `last_time` / `frame_count` Không Cập Nhật → FPS Sai ✅ ĐÃ SỬA

**File**: [main.py — dòng 44, 119-120](file:///c:/Users/Admin/BTL_XLA/testChess/main.py#L44)

**Vấn đề**: Code cập nhật `last_time`/`frame_count` bị comment → FPS luôn = 1.

**Cách sửa**: Dùng `start_time = time.time()` trước loop, `frame_count += 1` mỗi frame, FPS = `frame_count / (current_time - start_time)`.

---

## Bug 7: Contour Tọa Độ Gốc, Vẽ Lên Ảnh Resize ✅ ĐÃ SỬA

**File**: [main.py — dòng 90-102](file:///c:/Users/Admin/BTL_XLA/testChess/main.py#L90)

**Vấn đề**: `processor.last_board_contour` có tọa độ trên frame gốc, nhưng vẽ lên ảnh đã resize 640×480.

**Cách sửa**: Tính `scale_x`, `scale_y` từ kích thước gốc → display, nhân contour coordinates trước khi vẽ.

---

## Bug 8: Thiếu `import os` Trong `visualizer.py` ✅ ĐÃ SỬA

**File**: [visualizer.py — dòng 1](file:///c:/Users/Admin/BTL_XLA/testChess/visualizer.py#L1)

**Cách sửa**: Thêm `import os` ở đầu file.

---

## Tổng Hợp

| # | Bug | File đã sửa | Trạng thái |
|---|---|---|---|
| 1 | `prev_img` không reset khi undo | `move_detect.py` | ✅ Đã sửa |
| 2 | Castling/En Passant logic không gọi | `move_detect.py` | ✅ Đã sửa |
| 3 | `waitKey()` gọi 2 lần → mất phím | `main.py` | ✅ Đã sửa |
| 4 | Hardcoded video path | `main.py` | ✅ Đã sửa |
| 5 | `os.remove()` hardcoded crash | `main.py` | ✅ Đã sửa |
| 6 | FPS luôn = 1 | `main.py` | ✅ Đã sửa |
| 7 | Contour vẽ lệch sau resize | `main.py` | ✅ Đã sửa |
| 8 | Thiếu `import os` | `visualizer.py` | ✅ Đã sửa |

> ⚠️ **Lưu ý về Bug 1**: Dù đã sửa code, user vẫn phải **đặt lại quân trên bàn thật** trước khi nhấn `'i'` sau khi undo. Code chỉ giúp tránh crash/detect sai, nhưng không thể tự động đặt lại quân vật lý.
