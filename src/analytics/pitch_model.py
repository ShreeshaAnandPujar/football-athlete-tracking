import numpy as np
import cv2

class FootballPitch:
    """Standard FIFA Association Football Pitch Metric Model (105.0m x 68.0m)."""
    
    LENGTH = 105.0  # meters (X-axis: 0 to 105)
    WIDTH = 68.0    # meters (Y-axis: 0 to 68)
    
    # Key pitch landmarks in metric coordinates (X, Y)
    LANDMARKS = {
        # Outer pitch corners
        "corner_top_left": (0.0, 0.0),
        "corner_top_right": (105.0, 0.0),
        "corner_bottom_right": (105.0, 68.0),
        "corner_bottom_left": (0.0, 68.0),
        
        # Halfway line
        "halfway_top": (52.5, 0.0),
        "halfway_bottom": (52.5, 68.0),
        "center_spot": (52.5, 34.0),
        "center_circle_top": (52.5, 34.0 - 9.15),
        "center_circle_bottom": (52.5, 34.0 + 9.15),
        
        # Left penalty area (18-yard box: 16.5m deep, 40.32m wide)
        "left_penalty_top_left": (0.0, 34.0 - 20.16),
        "left_penalty_top_right": (16.5, 34.0 - 20.16),
        "left_penalty_bottom_right": (16.5, 34.0 + 20.16),
        "left_penalty_bottom_left": (0.0, 34.0 + 20.16),
        "left_penalty_spot": (11.0, 34.0),
        
        # Left goal area (6-yard box: 5.5m deep, 18.32m wide)
        "left_goal_top_left": (0.0, 34.0 - 9.16),
        "left_goal_top_right": (5.5, 34.0 - 9.16),
        "left_goal_bottom_right": (5.5, 34.0 + 9.16),
        "left_goal_bottom_left": (0.0, 34.0 + 9.16),
        
        # Right penalty area
        "right_penalty_top_right": (105.0, 34.0 - 20.16),
        "right_penalty_top_left": (105.0 - 16.5, 34.0 - 20.16),
        "right_penalty_bottom_left": (105.0 - 16.5, 34.0 + 20.16),
        "right_penalty_bottom_right": (105.0, 34.0 + 20.16),
        "right_penalty_spot": (105.0 - 11.0, 34.0),
        
        # Right goal area
        "right_goal_top_right": (105.0, 34.0 - 9.16),
        "right_goal_top_left": (105.0 - 5.5, 34.0 - 9.16),
        "right_goal_bottom_left": (105.0 - 5.5, 34.0 + 9.16),
        "right_goal_bottom_right": (105.0, 34.0 + 9.16),
    }

    @classmethod
    def draw_2d_pitch(cls, canvas_width=1050, canvas_height=680, padding=40, bg_color=(34, 110, 48), line_color=(255, 255, 255)):
        """
        Renders a high-resolution 2D tactical soccer pitch image in BGR.
        Returns:
            img: np.ndarray (canvas_height + 2*padding, canvas_width + 2*padding, 3)
            scale_x, scale_y: float conversion ratios from meters to canvas pixels.
        """
        full_w = canvas_width + 2 * padding
        full_h = canvas_height + 2 * padding
        img = np.full((full_h, full_w, 3), bg_color, dtype=np.uint8)
        
        scale_x = canvas_width / cls.LENGTH
        scale_y = canvas_height / cls.WIDTH
        
        def to_canvas(x_m, y_m):
            cx = int(padding + x_m * scale_x)
            cy = int(padding + y_m * scale_y)
            return (cx, cy)
        
        # Pitch Boundary
        p_tl = to_canvas(0, 0)
        p_br = to_canvas(cls.LENGTH, cls.WIDTH)
        cv2.rectangle(img, p_tl, p_br, line_color, 2)
        
        # Halfway line
        p_half_top = to_canvas(52.5, 0)
        p_half_bot = to_canvas(52.5, 68)
        cv2.line(img, p_half_top, p_half_bot, line_color, 2)
        
        # Center Circle & Center Spot
        c_spot = to_canvas(52.5, 34.0)
        cv2.circle(img, c_spot, int(round(9.15 * scale_y)), line_color, 2)
        cv2.circle(img, c_spot, 4, line_color, -1)
        
        # Left Penalty Box & Goal Box
        cv2.rectangle(img, to_canvas(0, 34.0 - 20.16), to_canvas(16.5, 34.0 + 20.16), line_color, 2)
        cv2.rectangle(img, to_canvas(0, 34.0 - 9.16), to_canvas(5.5, 34.0 + 9.16), line_color, 2)
        cv2.circle(img, to_canvas(11.0, 34.0), 3, line_color, -1)
        
        # Right Penalty Box & Goal Box
        cv2.rectangle(img, to_canvas(105.0 - 16.5, 34.0 - 20.16), to_canvas(105.0, 34.0 + 20.16), line_color, 2)
        cv2.rectangle(img, to_canvas(105.0 - 5.5, 34.0 - 9.16), to_canvas(105.0, 34.0 + 9.16), line_color, 2)
        cv2.circle(img, to_canvas(105.0 - 11.0, 34.0), 3, line_color, -1)
        
        # Penalty Arcs
        # Left arc
        cv2.ellipse(img, to_canvas(11.0, 34.0), (int(9.15 * scale_x), int(9.15 * scale_y)), 0, -53, 53, line_color, 2)
        # Right arc
        cv2.ellipse(img, to_canvas(105.0 - 11.0, 34.0), (int(9.15 * scale_x), int(9.15 * scale_y)), 0, 127, 233, line_color, 2)
        
        # Corner arcs
        r_corner = int(1.0 * scale_x)
        cv2.ellipse(img, to_canvas(0, 0), (r_corner, r_corner), 0, 0, 90, line_color, 2)
        cv2.ellipse(img, to_canvas(0, 68), (r_corner, r_corner), 0, 270, 360, line_color, 2)
        cv2.ellipse(img, to_canvas(105, 0), (r_corner, r_corner), 0, 90, 180, line_color, 2)
        cv2.ellipse(img, to_canvas(105, 68), (r_corner, r_corner), 0, 180, 270, line_color, 2)
        
        return img, scale_x, scale_y, padding
