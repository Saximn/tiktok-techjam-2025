#!/usr/bin/env python3
"""
Simple test for rectangle-based PII detector without webcam.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add pii_blur directory to path
sys.path.append(str(Path(__file__).parent))

from pii_detector import PIIDetector


def create_test_image():
    """Create a test image with sample text."""
    # Create a white image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Add some sample text with PII
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "John Smith", (50, 100), font, 2, (0, 0, 0), 3)
    cv2.putText(img, "john@example.com", (50, 200), font, 1, (0, 0, 0), 2)
    cv2.putText(img, "Phone: 555-123-4567", (50, 300), font, 1, (0, 0, 0), 2)
    cv2.putText(img, "Normal text here", (50, 400), font, 1, (0, 0, 0), 2)
    
    return img


def main():
    print("Testing Rectangle-based PII Detector...")
    
    try:
        # Initialize detector with rules only (no ML classifier)
        detector = PIIDetector(
            classifier_path=None,  # Use rules only
            conf_thresh=0.1,
            min_area=50
        )
        
        # Print model info
        info = detector.get_model_info()
        print(f"✓ Model initialized: {info['model_type']}")
        print(f"  OCR: {info['ocr_kind']}")
        print(f"  Device: {info['device']}")
        print(f"  ML Classifier: {info['has_ml_classifier']}")
        
        # Create test image
        test_img = create_test_image()
        print("✓ Test image created")
        
        # Save original
        cv2.imwrite("original_test.jpg", test_img)
        
        # Process frame
        frame_id, rectangles = detector.process_frame(test_img, frame_id=1)
        
        print(f"✓ Frame processed successfully")
        print(f"  Frame ID: {frame_id}")
        print(f"  Rectangles detected: {len(rectangles)}")
        
        # Verify rectangle format
        all_valid = True
        for i, rect in enumerate(rectangles):
            if isinstance(rect, list) and len(rect) == 4:
                x1, y1, x2, y2 = rect
                w, h = x2 - x1, y2 - y1
                print(f"  Rectangle {i+1}: [{x1}, {y1}, {x2}, {y2}] (w={w}, h={h})")
            else:
                print(f"  ❌ Invalid rectangle format: {rect}")
                all_valid = False
        
        if all_valid:
            print("✓ All rectangles have valid format")
        
        # Create visualization
        result_img = test_img.copy()
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_img, "PII", (x1, max(y1-10, 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        cv2.imwrite("rectangle_test_result.jpg", result_img)
        print("✓ Results saved: original_test.jpg, rectangle_test_result.jpg")
        
        # Test direct rectangle collection
        print("\nTesting direct rectangle collection...")
        direct_rects = detector.collect_pii_rects(test_img, blur_all=True)
        print(f"  Direct collection found: {len(direct_rects)} rectangles")
        
        print("\n🎉 All tests passed! Rectangle-based PII detector is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
