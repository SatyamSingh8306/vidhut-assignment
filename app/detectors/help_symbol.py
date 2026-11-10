import cv2
import numpy as np


class HelpSymbolDetector:
    """Detects if a person is asking for help using visual symbols/gestures.
    
    Symbols include:
    - Red help card / "HELP" sign
    - Raised open palm (help gesture)
    - Both hands raised above head
    - Specific color marker (red/yellow help symbol)
    """

    def __init__(self, confirm_frames: int = 3):
        self.confirm_frames = confirm_frames
        self.help_frames = 0
        self.current_symbol = None
        
    def update(self, largest_box, gesture_hint: str = None, frame: np.ndarray | None = None) -> tuple[str, float, bool]:
        """
        Detect help symbol / gesture.
        `largest_box` may be None, a list [x1,y1,x2,y2], or an ndarray image region.
        Returns: (symbol_type, confidence, is_help_requested)
        """
        # Check for help gestures first (works even without a frame)
        if gesture_hint in ("open_palm", "wave", "raised_hand", "open_hand"):
            self.help_frames += 1
            if self.help_frames >= self.confirm_frames:
                self.help_frames = 0
                return "help_gesture", 0.85, True
            return "none", 0.5, False  # building confirmation; do not decay
        if gesture_hint == "both_raised":
            self.help_frames += 1
            if self.help_frames >= self.confirm_frames:
                self.help_frames = 0
                return "both_raised", 0.9, True
            return "none", 0.5, False  # building confirmation; do not decay

        # Resolve image region from box or frame
        image_region = frame
        if image_region is None and isinstance(largest_box, np.ndarray):
            image_region = largest_box
        if image_region is None and frame is None and not isinstance(largest_box, (list, tuple)):
            image_region = largest_box

        if image_region is None or getattr(image_region, "size", 0) == 0:
            self.help_frames = 0
            return "none", 0.0, False
        
        # Check for red/yellow help card (visual symbol detection)
        # Convert to HSV and check for red/yellow dominant colors
        hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV) if len(image_region.shape) == 3 else cv2.cvtColor(image_region, cv2.COLOR_GRAY2BGR)
        
        # Red detection (help cards often red)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = np.sum(mask_red > 0)
        red_ratio = red_pixels / (image_region.shape[0] * image_region.shape[1]) if image_region.shape[0] > 0 else 0
        
        # Yellow help marker
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_pixels = np.sum(mask_yellow > 0)
        yellow_ratio = yellow_pixels / (image_region.shape[0] * image_region.shape[1]) if image_region.shape[0] > 0 else 0
        
        # Check if symbol is present (red help card or yellow marker)
        if red_ratio > 0.15 or yellow_ratio > 0.15:
            self.help_frames += 1
            self.current_symbol = "help_card" if red_ratio > yellow_ratio else "help_marker"
            if self.help_frames >= self.confirm_frames:
                self.help_frames = 0
                return self.current_symbol or "help_symbol", 0.9, True
        
        # Reset if no help signal
        self.help_frames = max(0, self.help_frames - 1)
        return "none", 0.0, False
    
    def get_current_symbol(self) -> tuple[str, float]:
        return self.current_symbol or "none", 0.0 if not self.help_frames else 0.8
