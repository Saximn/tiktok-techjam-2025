"""
Test script for the rectangle-based PII detector.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add pii_blur directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from pii_detector import PIIDetector
    
    def create_test_image():
        """Create a test image with sample text."""
        # Create a white image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Add some sample text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "John Smith", (50, 100), font, 2, (0, 0, 0), 3)
        cv2.putText(img, "john@example.com", (50, 200), font, 1, (0, 0, 0), 2)
        cv2.putText(img, "123 Main Street", (50, 300), font, 1, (0, 0, 0), 2)
        cv2.putText(img, "Normal text here", (50, 400), font, 1, (0, 0, 0), 2)
        
        return img
    
    def test_pii_detector():
        """Test the PII detector with webcam or test image."""
        print("Testing Rectangle-based PII Detector...")
        
        # Initialize detector
        detector = PIIDetector(
            classifier_path=None,  # Use rules only for testing
            conf_thresh=0.1,
            min_area=50,
            K_confirm=2,
            K_hold=8
        )
        
        # Print model info
        info = detector.get_model_info()
        print("Model Info:", info)
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam, using test image")
            # Create a test image
            frame = create_test_image()
            frame_id, rectangles = detector.process_frame(frame, 0)
            print(f"Frame {frame_id}: {len(rectangles)} rectangles to blur")
            for i, rect in enumerate(rectangles):
                print(f"  Rectangle {i}: {rect}")
                
            # Visualize
            vis_frame = frame.copy()
            for rect in rectangles:
                x1, y1, x2, y2 = rect
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, "PII", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv2.imwrite("test_pii_result.jpg", vis_frame)
            print("Result saved as test_pii_result.jpg")
            return
        
        frame_id = 0
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            returned_frame_id, rectangles = detector.process_frame(frame, frame_id)
            
            # Visualize
            vis_frame = frame.copy()
            for rect in rectangles:
                x1, y1, x2, y2 = rect
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, "PII", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(vis_frame, f"Frame: {frame_id}, PII regions: {len(rectangles)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("PII Detector Test", vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_id += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("PII detector test completed")

    if __name__ == "__main__":
        test_pii_detector()
        
except ImportError as e:
    print(f"PII detector not available: {e}")
    print("Make sure required packages are installed: pip install torch doctr easyocr scikit-learn")
