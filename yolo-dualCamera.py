import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time

class DualCameraYOLO:
    def __init__(self, model_path='best.pt', cam1_id=0, cam2_id=1):
        """
        Initialize dual camera YOLO detection system
        
        Args:
            model_path (str): Path to YOLO model file
            cam1_id (int): Camera 1 ID (usually 0 for built-in camera)
            cam2_id (int): Camera 2 ID (usually 1 for external camera)
        """
        self.model = YOLO(model_path)
        self.cam1_id = cam1_id
        self.cam2_id = cam2_id
        
        # Initialize cameras
        self.cap1 = cv2.VideoCapture(cam1_id)
        self.cap2 = cv2.VideoCapture(cam2_id)
        
        # Set camera properties (optional)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Queues for frame processing
        self.frame_queue1 = queue.Queue(maxsize=2)
        self.frame_queue2 = queue.Queue(maxsize=2)
        self.result_queue1 = queue.Queue(maxsize=2)
        self.result_queue2 = queue.Queue(maxsize=2)
        
        # Control flags
        self.running = False
        
        # Detection statistics
        self.fps_cam1 = 0
        self.fps_cam2 = 0
        self.detection_count_cam1 = 0
        self.detection_count_cam2 = 0

    def capture_frames(self, cap, frame_queue, cam_name):
        """Capture frames from camera in separate thread"""
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Keep only the latest frame
                if not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)
            time.sleep(0.01)  # Small delay to prevent overwhelming

    def process_detection(self, frame_queue, result_queue, cam_name):
        """Process YOLO detection in separate thread"""
        prev_time = time.time()
        frame_count = 0
        
        while self.running:
            try:
                frame = frame_queue.get(timeout=0.1)
                
                # Run YOLO detection
                results = self.model(frame)
                
                # Process results
                annotated_frame = results[0].plot()
                
                # Count detections
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - prev_time >= 1.0:
                    fps = frame_count / (current_time - prev_time)
                    if cam_name == "Camera 1":
                        self.fps_cam1 = fps
                        self.detection_count_cam1 = detections
                    else:
                        self.fps_cam2 = fps
                        self.detection_count_cam2 = detections
                    
                    frame_count = 0
                    prev_time = current_time
                
                # Add info overlay
                self.add_info_overlay(annotated_frame, cam_name, detections)
                
                # Keep only the latest result
                if not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass
                result_queue.put(annotated_frame)
                
            except queue.Empty:
                continue

    def add_info_overlay(self, frame, cam_name, detection_count):
        """Add information overlay to frame"""
        fps = self.fps_cam1 if cam_name == "Camera 1" else self.fps_cam2
        
        # Background rectangle for text
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        
        # Add text information
        cv2.putText(frame, f"{cam_name}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Main execution function"""
        if not self.cap1.isOpened():
            print(f"Error: Cannot open camera {self.cam1_id}")
            return
        
        if not self.cap2.isOpened():
            print(f"Error: Cannot open camera {self.cam2_id}")
            return
        
        print("Starting dual camera YOLO detection...")
        print("Press 'q' to quit, 's' to save screenshots")
        
        self.running = True
        
        # Start capture threads
        thread1 = threading.Thread(target=self.capture_frames, 
                                 args=(self.cap1, self.frame_queue1, "Camera 1"))
        thread2 = threading.Thread(target=self.capture_frames, 
                                 args=(self.cap2, self.frame_queue2, "Camera 2"))
        
        # Start processing threads
        process_thread1 = threading.Thread(target=self.process_detection, 
                                         args=(self.frame_queue1, self.result_queue1, "Camera 1"))
        process_thread2 = threading.Thread(target=self.process_detection, 
                                         args=(self.frame_queue2, self.result_queue2, "Camera 2"))
        
        # Start all threads
        thread1.start()
        thread2.start()
        process_thread1.start()
        process_thread2.start()
        
        # Main display loop
        while self.running:
            try:
                # Get processed frames
                frame1 = self.result_queue1.get_nowait() if not self.result_queue1.empty() else None
                frame2 = self.result_queue2.get_nowait() if not self.result_queue2.empty() else None
                
                # Display frames
                if frame1 is not None:
                    cv2.imshow('Camera 1 - YOLO Detection', frame1)
                
                if frame2 is not None:
                    cv2.imshow('Camera 2 - YOLO Detection', frame2)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    self.running = False
                    break
                elif key == ord('s'):
                    # Save screenshots
                    if frame1 is not None:
                        timestamp = int(time.time())
                        cv2.imwrite(f'camera1_detection_{timestamp}.jpg', frame1)
                        print(f"Screenshot saved: camera1_detection_{timestamp}.jpg")
                    
                    if frame2 is not None:
                        timestamp = int(time.time())
                        cv2.imwrite(f'camera2_detection_{timestamp}.jpg', frame2)
                        print(f"Screenshot saved: camera2_detection_{timestamp}.jpg")
                
            except queue.Empty:
                continue
        
        # Clean up
        self.cleanup()
        
        # Wait for threads to complete
        thread1.join()
        thread2.join()
        process_thread1.join()
        process_thread2.join()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.cap1.release()
        self.cap2.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

    def get_statistics(self):
        """Get detection statistics"""
        return {
            'camera1_fps': self.fps_cam1,
            'camera2_fps': self.fps_cam2,
            'camera1_detections': self.detection_count_cam1,
            'camera2_detections': self.detection_count_cam2
        }


def main():
    """Main function to run dual camera detection"""
    try:
        print("Detecting available cameras...")
        
        # Initialize the dual camera system with auto-detection
        detector = DualCameraYOLO(
            model_path='best.pt'  # Path to your YOLO model
        )
        
        # Run the detection system
        detector.run()
        
        # Print final statistics
        stats = detector.get_statistics()
        print("\nFinal Statistics:")
        print(f"Camera 1 - FPS: {stats['camera1_fps']:.1f}, Last Detection Count: {stats['camera1_detections']}")
        print(f"Camera 2 - FPS: {stats['camera2_fps']:.1f}, Last Detection Count: {stats['camera2_detections']}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except FileNotFoundError:
        print("Error: 'best.pt' model file not found!")
        print("Please make sure your YOLO model file is in the correct path.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure at least one camera is connected")
        print("2. Check if camera is being used by another application")
        print("3. Try running as administrator")
        print("4. Make sure 'best.pt' model file exists")


if __name__ == "__main__":
    main()
