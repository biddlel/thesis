import cv2
import numpy as np
import argparse
import os

class MicrophoneArrayCalibrator:
    def __init__(self):
        """Initialize the frame-based calibrator"""
        # Store points
        self.frame_corners = []
        self.mic_points = []
        self.homography = None
    
    def select_points(self, img):
        """Select frame corners and microphones"""
        points = []
        img_copy = img.copy()
        
        def click_event(event, x, y, flags, param):
            nonlocal img_copy
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 8:  # 4 frame corners + 4 mics
                    points.append((x, y))
                    if len(points) <= 4:
                        # Frame corners
                        cv2.circle(img_copy, (x, y), 8, (0, 0, 255), -1)
                        cv2.putText(img_copy, f"Frame {len(points)}", (x+10, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Microphones
                        cv2.circle(img_copy, (x, y), 8, (255, 0, 0), -1)
                        cv2.putText(img_copy, f"Mic {len(points)-4}", (x+10, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow('Select Points', img_copy)
        
        print("\nClick on the points in this order:")
        print("1. Frame corner 1 (bottom-left)")
        print("2. Frame corner 2 (bottom-right)")
        print("3. Frame corner 3 (top-right)")
        print("4. Frame corner 4 (top-left)")
        print("5. Mic 1 (bottom-left)")
        print("6. Mic 2 (bottom-right)")
        print("7. Mic 3 (top-left)")
        print("8. Mic 4 (top-right)")
        print("Then press 'q' when done")
        
        cv2.imshow('Select Points', img_copy)
        cv2.setMouseCallback('Select Points', click_event)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or len(points) == 8:
                break
        
        cv2.destroyAllWindows()
        
        if len(points) == 8:
            self.frame_corners = np.array(points[:4], dtype=np.float32)
            self.mic_points = np.array(points[4:], dtype=np.float32)
            return True
        return False
    
    def calculate_homography(self, frame_width_mm, frame_height_mm):
        """Calculate homography using frame dimensions"""
        if len(self.frame_corners) != 4:
            return False
            
        # Define destination points based on actual frame dimensions
        dst_pts = np.array([
            [0, frame_height_mm],          # bottom-left
            [frame_width_mm, frame_height_mm],  # bottom-right
            [frame_width_mm, 0],           # top-right
            [0, 0]                         # top-left
        ], dtype=np.float32)
        
        # Calculate homography
        self.homography, _ = cv2.findHomography(self.frame_corners, dst_pts)
        return self.homography is not None
    
    def transform_points(self):
        """Transform microphone points to frame coordinates"""
        if self.homography is None or len(self.mic_points) != 4:
            return False
            
        # Transform points
        self.transformed_mics = cv2.perspectiveTransform(
            self.mic_points.reshape(1, -1, 2), self.homography
        )[0]
        
        # Make bottom-left corner (0,0)
        min_x = np.min(self.transformed_mics[:, 0])
        min_y = np.min(self.transformed_mics[:, 1])
        self.transformed_mics -= [min_x, min_y]
        
        return True
    
    def print_results(self):
        """Print the calculated microphone positions"""
        if not hasattr(self, 'transformed_mics'):
            print("No microphone positions calculated")
            return
        
        print("\nMicrophone positions (in mm, relative to frame corner):")
        print("{")
        for i, (x, y) in enumerate(self.transformed_mics):
            print(f"  {{ {x:8.2f}, {y:8.2f}, 0.0 }},  // Mic {i+1}")
        print("}")
        
        # Print as C++ array for easy copy-paste
        print("\nC++ array for Teensy code:")
        print("MicCoordinate mic_coords[4] = {")
        for i, (x, y) in enumerate(self.transformed_mics):
            comma = "," if i < 3 else ""
            print(f"  {{ {x:8.2f}, {y:8.2f}, 0.0 }}{comma}  // Mic {i+1}")
        print("};")

def main():
    parser = argparse.ArgumentParser(description='Microphone Array Frame-based Calibration Tool')
    parser.add_argument('image_path', type=str, help='Path to the image of the microphone array and frame')
    parser.add_argument('--width', type=float, default=204, help='Frame width in mm')
    parser.add_argument('--height', type=float, default=200, help='Frame height in mm')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return
    
    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not load image '{args.image_path}'")
        return
    
    # Initialize calibrator
    calibrator = MicrophoneArrayCalibrator()
    
    # Select frame corners and microphones
    print("\nSelect points in order (see terminal for instructions)")
    if not calibrator.select_points(img):
        print("Error: Failed to select all points")
        return
    
    # Calculate homography using frame dimensions
    if not calibrator.calculate_homography(args.width, args.height):
        print("Error: Failed to calculate perspective transform")
        return
    
    # Transform points to frame coordinates
    if not calibrator.transform_points():
        print("Error: Failed to transform points")
        return
    
    # Print results
    calibrator.print_results()
    
    # Show final result with points
    result_img = img.copy()
    
    # Draw frame
    for i, (x, y) in enumerate(calibrator.frame_corners):
        cv2.circle(result_img, (int(x), int(y)), 10, (0, 0, 255), -1)
        cv2.putText(result_img, f"Frame {i+1}", (int(x)+15, int(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw microphones
    for i, (x, y) in enumerate(calibrator.mic_points):
        cv2.circle(result_img, (int(x), int(y)), 10, (255, 0, 0), -1)
        cv2.putText(result_img, f"Mic {i+1}", (int(x)+15, int(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show perspective-corrected view
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, calibrator.homography, (w, h))
    
    # Draw transformed points on warped image
    for i, (x, y) in enumerate(calibrator.transformed_mics):
        # Scale points back to image coordinates for display
        px = int(x * (w/args.width))
        py = int((args.height - y) * (h/args.height))  # Flip y-axis
        cv2.circle(warped, (px, py), 15, (0, 255, 0), 3)
        cv2.putText(warped, f"Mic {i+1}", (px+20, py+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Original with Points', cv2.resize(result_img, (800, 600)))
    cv2.imshow('Perspective Corrected', cv2.resize(warped, (800, 600)))
    print("\nCalibration complete! Close windows to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
