# FILE: submain.py (THE CLIENT)
import cv2
import requests
import time

# UPDATE THIS IP IF RUNNING ON DIFFERENT MACHINES
# Example: "http://192.168.1.100:5055/compare/"
API_URL = "http://localhost:5055/compare/"

def start_camera_client():
    # 1. Ask user for the Product ID to check
    target_id = input("Enter Product ID to Validate (or press Enter to just scan): ").strip()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("------------------------------------------------")
    print(f" TARGET ID: {target_id if target_id else 'ANY (Guessing Mode)'}")
    print(" [SPACE] : Snap & Compare")
    print(" [Q]     : Quit")
    print("------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # IMPORTANT: Make a copy for display so we don't draw on the AI image
        display_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Draw Green Box (Visual Guide Only)
        cv2.rectangle(display_frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        
        # Show ID on screen
        cv2.putText(display_frame, f"Target: {target_id}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Webcam Client", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Space Bar
            print("\nCapturing image...")
            
            # SEND THE ORIGINAL FRAME (Clean, no green lines)
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            try:
                t0 = time.time()
                
                # Send Image + Target ID to Server
                response = requests.post(
                    API_URL, 
                    files={"file": ("capture.jpg", img_encoded.tobytes(), "image/jpeg")},
                    data={"expected_id": target_id} if target_id else None
                )
                t1 = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    pid = data.get("product_id", "Unknown")
                    dist = data.get("confidence_score", 0.0)
                    msg = data.get("message", "")

                    if data.get("match"):
                        print(f"✅ MATCH: {pid} (Conf: {dist:.4f} | {t1-t0:.2f}s)")
                        if "VERIFIED" in msg: print("   Verified against Target ID.")
                    else:
                        print(f"❌ FAILED: {msg} (Dist: {dist:.4f})")
                else:
                    print(f"Server Error: {response.text}")

            except Exception as e:
                print(f"Connection Error: {e}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_client()