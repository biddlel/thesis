import cv2
import numpy as np
import argparse
import os

def create_zoomed_view(img, center, zoom=4, window_size=200):
    """Create a zoomed-in view around the center point"""
    h, w = img.shape[:2]
    half_win = window_size // 2
    
    # Calculate crop coordinates
    x1 = max(0, center[0] - half_win)
    y1 = max(0, center[1] - half_win)
    x2 = min(w, center[0] + half_win)
    y2 = min(h, center[1] + half_win)
    
    # Crop and zoom
    cropped = img[y1:y2, x1:x2]
    if cropped.size > 0:
        zoomed = cv2.resize(cropped, (window_size*2, window_size*2), 
                          interpolation=cv2.INTER_LINEAR)
        # Draw crosshair
        cv2.line(zoomed, (window_size, 0), (window_size, window_size*2), (0, 255, 0), 1)
        cv2.line(zoomed, (0, window_size), (window_size*2, window_size), (0, 255, 0), 1)
        return zoomed
    return np.zeros((window_size*2, window_size*2, 3), dtype=np.uint8)

class MicrophoneArrayCalibrator:
    def __init__(self):
        """Initialize the calibrator"""
        self.frame_corners = []  # Will store 4 points
        self.mic_points = []     # Will store 4 points
        self.origin = None       # Will store the origin point
        self.homography = None
        self.transformed_mics = None

    def select_points(self, img_path, frame_width_mm, frame_height_mm):
        """Main function to handle point selection and calibration"""
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            return False

        # Create a copy for drawing
        img_copy = img.copy()
        zoom_win = 'Zoomed View'
        
        # Create resizable windows
        cv2.namedWindow('Setup Frame and Mics', cv2.WINDOW_NORMAL)
        cv2.namedWindow(zoom_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(zoom_win, 400, 400)

        # First, select the origin point (first point)
        print("\n=== Point Selection ===")
        print("1. First, select the origin point (0,0)")
        print("2. Then select the 4 frame corners in order:")
        print("   1. Bottom-left corner (min X, min Y)")
        print("   2. Bottom-right corner (max X, min Y)")
        print("   3. Top-right corner (max X, max Y)")
        print("   4. Top-left corner (min X, max Y)")
        print("3. Finally, select the 4 microphone positions in the same order:")
        print("   1. Bottom-left mic (min X, min Y)")
        print("   2. Bottom-right mic (max X, min Y)")
        print("   3. Top-right mic (max X, max Y)")
        print("   4. Top-left mic (min X, max Y)")
        print("\nClick to select each point. Right-click to undo last point.")
        print("Press 'q' when done.")

        def redraw_all_points():
            nonlocal img_copy
            # Create a fresh copy of the image
            img_copy = img.copy()
            
            # Draw origin if it exists and is not None
            if self.origin is not None:
                ox, oy = int(self.origin[0]), int(self.origin[1])
                cv2.drawMarker(img_copy, (ox, oy), (0, 255, 0), 
                             cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img_copy, "Origin (0,0)", (ox+10, oy-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw frame corners
            for i, (px, py) in enumerate(self.frame_corners):
                cv2.circle(img_copy, (px, py), 8, (0, 0, 255), -1)  # Red for frame
                cv2.putText(img_copy, f"Frame {i+1}", (px+15, py-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw microphone points
            for i, (px, py) in enumerate(self.mic_points):
                cv2.circle(img_copy, (px, py), 8, (255, 0, 0), -1)  # Blue for mics
                cv2.putText(img_copy, f"Mic {i+1}", (px+15, py-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Update the display
            cv2.imshow('Setup Frame and Mics', img_copy)
            
            # Update zoom window if mouse is over the image
            if 'x' in locals() and 'y' in locals():
                zoomed = create_zoomed_view(img_copy, (x, y))
                cv2.imshow(zoom_win, zoomed)
        
        def setup_click_event(event, x, y, flags, param):
            nonlocal img_copy
            
            # Update zoom window on mouse move
            if event == cv2.EVENT_MOUSEMOVE:
                redraw_all_points()
                zoomed = create_zoomed_view(img_copy, (x, y))
                cv2.imshow(zoom_win, zoomed)
            
            # Handle left click - add point
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.origin is None:
                    # First point is the origin
                    self.origin = np.array([x, y], dtype=np.float32)
                    print("\nOrigin set. Now select the 4 frame corners.")
                    
                elif len(self.frame_corners) < 4:
                    # Add frame corner
                    self.frame_corners.append((x, y))
                    # If we just finished frame corners, switch to mic points
                    if len(self.frame_corners) == 4:
                        print("\nNow select the 4 microphone positions.")
                
                # Add microphone points after frame corners are done
                elif len(self.mic_points) < 4:
                    self.mic_points.append((x, y))
                
                # Redraw all points
                redraw_all_points()
            
            # Handle right click - undo last point
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.mic_points) > 0:
                    # Undo last mic point
                    self.mic_points.pop()
                elif len(self.frame_corners) > 0:
                    # Undo last frame corner
                    self.frame_corners.pop()
                elif hasattr(self, 'origin'):
                    # Remove origin if it exists
                    delattr(self, 'origin')
                
                # Redraw all points
                img_copy = img.copy()
                
                # Redraw origin if it exists
                if hasattr(self, 'origin'):
                    px, py = int(self.origin[0]), int(self.origin[1])
                    cv2.drawMarker(img_copy, (px, py), (0, 255, 0), 
                                 cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(img_copy, "Origin (0,0)", (px+10, py-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Redraw frame corners
                for i, (px, py) in enumerate(self.frame_corners):
                    cv2.circle(img_copy, (px, py), 8, (0, 0, 255), -1)
                    cv2.putText(img_copy, f"Frame {i+1}", (px+15, py-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Redraw mic points
                for i, (px, py) in enumerate(self.mic_points):
                    cv2.circle(img_copy, (px, py), 8, (255, 0, 0), -1)
                    cv2.putText(img_copy, f"Mic {i+1}", (px+15, py-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Redraw all points
                img_copy = img.copy()
                # Redraw frame corners
                for i, (px, py) in enumerate(self.frame_corners):
                    cv2.circle(img_copy, (px, py), 8, (0, 0, 255), -1)
                    cv2.putText(img_copy, f"Frame {i+1}", (px+15, py-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Redraw mic points
                for i, (px, py) in enumerate(self.mic_points):
                    cv2.circle(img_copy, (px, py), 8, (255, 0, 0), -1)
                    cv2.putText(img_copy, f"Mic {i+1}", (px+15, py-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # Redraw origin if it exists
                if hasattr(self, 'origin'):
                    px, py = int(self.origin[0]), int(self.origin[1])
                    cv2.drawMarker(img_copy, (px, py), (0, 255, 0), 
                                 cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(img_copy, "Origin (0,0)", (px+10, py-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update display
            cv2.imshow('Setup Frame and Mics', img_copy)
            if len(self.frame_corners) > 0 or len(self.mic_points) > 0 or hasattr(self, 'origin'):
                zoomed = create_zoomed_view(img_copy, (x, y))
                cv2.imshow(zoom_win, zoomed)

        # Set up mouse callback
        cv2.imshow('Setup Frame and Mics', img_copy)
        cv2.setMouseCallback('Setup Frame and Mics', setup_click_event)

        # Wait for all points to be selected
        while len(self.frame_corners) < 4 or len(self.mic_points) < 4 or not hasattr(self, 'origin'):
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False

        # Calculate homography
        # Define destination points with proper coordinate system:
        # Define destination points for homography
        # - X increases to the right
        # - Y increases upward
        # - Origin is at the center of the frame
        half_w = frame_width_mm / 2
        half_h = frame_height_mm / 2
        dst_pts = np.array([
            [-half_w, -half_h],    # bottom-left (min X, min Y)
            [ half_w, -half_h],    # bottom-right (max X, min Y)
            [ half_w,  half_h],    # top-right (max X, max Y)
            [-half_w,  half_h]     # top-left (min X, max Y)
        ], dtype=np.float32)
        
        self.homography, _ = cv2.findHomography(
            np.array(self.frame_corners, dtype=np.float32), 
            dst_pts
        )
        
        if self.homography is None:
            print("Error: Could not calculate homography")
            return False

        # Transform mic points to real-world coordinates
        mic_points_np = np.array(self.mic_points, dtype=np.float32).reshape(1, -1, 2)
        self.transformed_mics = cv2.perspectiveTransform(
            mic_points_np, self.homography
        )[0]
        
        # Get the origin point (either selected by user or use center of frame)
        if hasattr(self, 'origin'):
            origin_pt = np.array([self.origin[0], self.origin[1], 1])
        else:
            # If no origin selected, use the center of the frame
            frame_center = np.mean(np.array(self.frame_corners), axis=0)
            origin_pt = np.array([frame_center[0], frame_center[1], 1])
        
        # Transform origin to real-world coordinates
        origin_transformed = np.dot(self.homography, origin_pt)
        origin_x = origin_transformed[0] / origin_transformed[2]
        origin_y = origin_transformed[1] / origin_transformed[2]
        
        # Adjust mic positions relative to origin
        self.transformed_mics = [(x - origin_x, y - origin_y) 
                               for x, y in self.transformed_mics]
        
        # Create a visualization of the warped image with mic positions
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, self.homography, 
                                   (int(frame_width_mm), int(frame_height_mm)))
        
        # Draw the origin and mic positions on the warped image
        vis_img = warped.copy()
        
        # Draw the origin
        ox, oy = int(origin_x), int(origin_y)
        cv2.drawMarker(vis_img, (ox, oy), (0, 0, 255), 
                     cv2.MARKER_CROSS, 30, 2)
        cv2.putText(vis_img, "Origin (0,0)", (ox+10, oy-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw microphone positions
        for i, (mx, my) in enumerate(self.transformed_mics):
            # Convert back to warped image coordinates for visualization
            px, py = int(mx + ox), int(my + oy)
            cv2.circle(vis_img, (px, py), 10, (255, 0, 0), -1)
            cv2.putText(vis_img, f"Mic {i+1}", (px+15, py-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Show the final result
        cv2.namedWindow('Calibration Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Calibration Result', vis_img)
        
        # Show zoomed view of the origin
        zoomed = create_zoomed_view(vis_img, (ox, oy))
        cv2.imshow(zoom_win, zoomed)
        
        # Wait for a key press
        print("\nCalibration complete! Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True

    def print_results(self):
        """Print the calculated microphone positions with descriptive labels"""
        if self.transformed_mics is None:
            print("No microphone positions calculated")
            return
        
        # Define position labels in the order they were selected
        position_labels = [
            "Bottom-left",
            "Bottom-right",
            "Top-left",
            "Top-right"
        ]
        
        print("\n=== Calibration Results ===")
        print("Microphone positions (in mm, relative to origin):")
        print("{")
        for i, ((x, y), label) in enumerate(zip(self.transformed_mics, position_labels)):
            print(f"  {{ {x:8.2f}, {y:8.2f}, 0.0 }},  // {label} (Mic {i+1})")
        print("}")
        
        # Print as C++ array for easy copy-paste
        print("\nC++ array for Teensy code:")
        print("const float mic_positions[4][3] = {")
        for i, ((x, y), label) in enumerate(zip(self.transformed_mics, position_labels)):
            comma = "," if i < 3 else ""
            print(f"  {{ {x:8.2f}, {y:8.2f}, 0.0 }}{comma}  // {label} (Mic {i+1})")
        print("};")
        
        # Print a summary with coordinates and labels
        print("\nMicrophone Positions Summary:")
        print("-" * 40)
        for (x, y), label in zip(self.transformed_mics, position_labels):
            print(f"{label + ':':<12} X: {x:8.2f} mm, Y: {y:8.2f} mm")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='Microphone Array Calibration Tool')
    parser.add_argument('image_path', type=str, 
                       help='Path to the image of the microphone array and frame')
    parser.add_argument('--width', type=float, default=200.0, 
                       help='Frame width in mm (default: 204.0)')
    parser.add_argument('--height', type=float, default=204.0,
                       help='Frame height in mm (default: 200.0)')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return
    
    # Initialize and run calibrator
    calibrator = MicrophoneArrayCalibrator()
    if calibrator.select_points(args.image_path, args.width, args.height):
        calibrator.print_results()
    else:
        print("Calibration was not completed.")

if __name__ == "__main__":
    main()