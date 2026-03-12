# ♟️ CV Chess – Real-Time Chess Move Detection with OpenCV

Hệ thống nhận diện nước đi cờ vua từ camera bằng xử lý ảnh và thư viện `python-chess`.

Project gồm 2 chế độ hoạt động:

- `test_chess.py` → Tự động phát hiện bàn cờ bằng Contour Analysis (findContours)
- `test_chess_manualConner.py` → Hiệu chỉnh thủ công 4 góc (ổn định hơn)

---

# 🧠 Kiến trúc hệ thống

## 1. Phát hiện bàn cờ

Hệ thống sử dụng:

- Canny Edge Detection
- Contour Analysis (findContours)
- Xấp xỉ đa giác để tìm 4 góc (approxPolyDP)
- `cv2.getPerspectiveTransform`
- Warp về góc nhìn top-down 8x8

---

## 2. Phát hiện nước đi

Quy trình:

1. Lưu ảnh bàn cờ trước khi đi
2. So sánh với ảnh sau khi đi
3. Tính `absdiff`
4. Threshold nhị phân
5. Đếm pixel thay đổi theo từng ô
6. So khớp với nước đi hợp lệ từ thư viện `python-chess`

---

# 📦 Yêu cầu cài đặt

```bash
pip install opencv-python numpy python-chess
