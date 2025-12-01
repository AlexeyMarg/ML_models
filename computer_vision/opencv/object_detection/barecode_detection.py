import cv2
from pyzbar import pyzbar

# --- Initialize camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    exit(1)

def decode_barcodes(frame):
    """Detect and decode barcodes/QR codes on the frame."""
    barcodes = pyzbar.decode(frame)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        text = f"{barcode_type}: {barcode_data}"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        print(f"[FOUND] {barcode_type}: {barcode_data}")

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read frame")
        break

    frame = decode_barcodes(frame)
    cv2.imshow("Barcode & QR Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
