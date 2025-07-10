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
        self.frame_corners = []
        self.mic_points = []
        self.homography = None
        self.origin = None
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

        # First, select frame corners
        print("\n=== Frame Corner Selection ===")
        print("Please select the 4 corners of the frame in order:")
        print("1. Bottom-left corner")
        print("2. Bottom-right corner")
        print("3. Top-right corner")
        print("4. Top-left corner")
        print("\nClick to select each point. Right-click to undo last point.")
        print("Press 'q' when done.")

        def setup_click_event(event, x, y, flags, param):
            nonlocal img_copy
            
            # Update zoom window on mouse move
            if event == cv2.EVENT_MOUSEMOVE:
                zoomed = create_zoomed_view(img_copy, (x, y))
                cv2.imshow(zoom_win, zoomed)
            
            # Handle left click - add point
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.frame_corners) < 4:
                    # Only add frame corner if we haven't completed them yet
                    self.frame_corners.append((x, y))
                    # Draw the point
                    cv2.circle(img_copy, (x, y), 8, (0, 0, 255), -1)  # Red for frame
                    cv2.putText(img_copy, f"Frame {len(self.frame_corners)}", 
                              (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2)
                    
                    # If we just finished frame corners, switch to mic points
                    if len(self.frame_corners) == 4:
                        print("\n=== Microphone Selection ===")
                        print("Now select the 4 microphone positions:")
                        print("1. Bottom-left mic")
                        print("2. Bottom-right mic")
                        print("3. Top-left mic")
                        print("4. Top-right mic")
                
                # Only start adding mic points after all frame corners are selected
                elif len(self.frame_corners) == 4 and len(self.mic_points) < 4:
                    self.mic_points.append((x, y))
                    # Draw the point
                    cv2.circle(img_copy, (x, y), 8, (255, 0, 0), -1)  # Blue for mics
                    cv2.putText(img_copy, f"Mic {len(self.mic_points)}", 
                              (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 0, 0), 2)
            
            # Handle right click - undo last point
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.mic_points) > 0:
                    # Undo last mic point if any exist
                    self.mic_points.pop()
                elif len(self.frame_corners) > 0:
                    # Otherwise undo last frame corner
                    self.frame_corners.pop()
                
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
            
            # Update display
            cv2.imshow('Setup Frame and Mics', img_copy)
            if len(self.frame_corners) > 0 or len(self.mic_points) > 0:
                zoomed = create_zoomed_view(img_copy, (x, y))
                cv2.imshow(zoom_win, zoomed)

        # Set up mouse callback
        cv2.imshow('Setup Frame and Mics', img_copy)
        cv2.setMouseCallback('Setup Frame and Mics', setup_click_event)

        # Wait for all points to be selected
        while len(self.frame_corners) < 4 or len(self.mic_points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False

        # Calculate homography
        dst_pts = np.array([
            [0, frame_height_mm],          # bottom-left
            [frame_width_mm, frame_height_mm],  # bottom-right
            [frame_width_mm, 0],           # top-right
            [0, 0]                         # top-left
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

        # Now let the user select the origin
        print("\n=== Origin Selection ===")
        print("Select the origin point (0,0) on the warped image")
        print("Microphone positions will be calculated relative to this point")
        
        # Create a warped view for origin selection
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, self.homography, 
                                   (int(frame_width_mm), int(frame_height_mm)))
        
        origin_img = warped.copy()
        origin_selected = [False]

        def origin_click_event(event, x, y, flags, param):
            nonlocal origin_img, warped
            
            # Update zoom window on mouse move
            if event == cv2.EVENT_MOUSEMOVE and not origin_selected[0]:
                zoomed = create_zoomed_view(origin_img, (x, y))
                cv2.imshow(zoom_win, zoomed)
            
            # Handle origin selection
            if event == cv2.EVENT_LBUTTONDOWN and not origin_selected[0]:
                self.origin = np.array([x, y], dtype=np.float32)
                origin_img = warped.copy()
                
                # Draw crosshair at origin
                cv2.drawMarker(origin_img, (x, y), (0, 0, 255), 
                             cv2.MARKER_CROSS, 30, 2)
                cv2.putText(origin_img, "Origin (0,0)", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw microphone positions
                for i, (mx, my) in enumerate(self.transformed_mics):
                    px, py = int(mx), int(my)
                    cv2.circle(origin_img, (px, py), 10, (255, 0, 0), -1)
                    cv2.putText(origin_img, f"Mic {i+1}", (px+15, py-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow('Select Origin', origin_img)
                zoomed = create_zoomed_view(origin_img, (x, y))
                cv2.imshow(zoom_win, zoomed)
                origin_selected[0] = True

        cv2.namedWindow('Select Origin', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Origin', warped)
        cv2.setMouseCallback('Select Origin', origin_click_event)

        # Show initial zoomed view
        zoomed = create_zoomed_view(warped, 
                                  (warped.shape[1]//2, warped.shape[0]//2))
        cv2.imshow(zoom_win, zoomed)

        # Wait for origin selection
        while not origin_selected[0]:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False

        # Calculate final coordinates relative to origin
        self.transformed_mics -= self.origin
        
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