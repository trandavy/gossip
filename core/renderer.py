import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Renderer:
    def __init__(self, alpha=0.9):
        """
        Takes raw bounding boxes, smooths them out using moving averages,
        and renders alpha-blended speech bubbles pointing to targets using Pillow.
        """
        self.alpha = alpha
        
        # Load fonts. Fallback gracefully to default if standard Windows fonts are missing.
        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
        except OSError:
            try:
                self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
            except OSError:
                print("Warning: Arial font missing. Relying on basic default PIL font.")
                self.font = ImageFont.load_default()
                
        # Moving averages for smooth x, y bounding box centers
        self.ema_positions = {}  # tid: [x, y]
        # Hyperparameter determining smoothing strength (0.0 to 1.0)
        self.ema_alpha = 0.25

    def render(self, frame, tracks, actors, quotes):
        """
        Blend the PIL generated text strings and speech bubbles onto the final cv2 frame.
        """
        current_ids = set(tracks.keys())
        
        # Garbage collection for EMA states of inactive tracks
        for tid in list(self.ema_positions.keys()):
            if tid not in current_ids:
                del self.ema_positions[tid]
                
        # 1. Coordinate calculation and smoothing via EMA (Exponential Moving Average)
        for tid, tdata in tracks.items():
            x1, y1, x2, y2 = tdata['box']
            tx = (x1 + x2) / 2
            ty = y1
            
            if tid not in self.ema_positions:
                self.ema_positions[tid] = [tx, ty] # Start raw
            else:
                self.ema_positions[tid][0] = self.ema_alpha * tx + (1 - self.ema_alpha) * self.ema_positions[tid][0]
                self.ema_positions[tid][1] = self.ema_alpha * ty + (1 - self.ema_alpha) * self.ema_positions[tid][1]

        # 2. Rendering Layers
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        
        # We pre-allocate a transparent overlay to hold all the drawn speech bubbles
        overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # 3. Draw bubbles for chosen Actors
        for tid in actors:
            if tid in quotes and tid in self.ema_positions:
                text = quotes[tid]
                cx, cy = self.ema_positions[tid]
                self._draw_speech_bubble(draw, text, cx, cy)

        # 4. Composite final visual
        out_pil = Image.alpha_composite(img_pil, overlay)
        return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGBA2BGR)

    def _draw_speech_bubble(self, draw, text, cx, cy):
        """
        Private rendering sub-routine handling bubble aesthetic rules.
        """
        # Calculate dynamic text bounding box depending on length
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        pad_x, pad_y = 16, 12
        bubble_w = text_w + pad_x * 2
        bubble_h = text_h + pad_y * 2
        
        offset_y = 20  # Hover distance from true bounding box ceiling
        bubble_x1 = cx - bubble_w / 2
        bubble_y1 = cy - offset_y - bubble_h
        bubble_x2 = cx + bubble_w / 2
        bubble_y2 = cy - offset_y
        
        # Color palettes
        fill_color = (255, 255, 255, int(255 * self.alpha))
        shadow_color = (0, 0, 0, int(255 * 0.3))
        
        # Draw background shadow drop
        radius = 10
        draw.rounded_rectangle((bubble_x1+2, bubble_y1+2, bubble_x2+2, bubble_y2+2), radius, fill=shadow_color)
        
        # Draw pure white container box
        draw.rounded_rectangle((bubble_x1, bubble_y1, bubble_x2, bubble_y2), radius, fill=fill_color)
        
        # Draw comic-book arrow "tail" pointing to target
        tail_w, tail_h = 16, 16
        draw.polygon([
            (cx - tail_w / 2, bubble_y2), 
            (cx + tail_w / 2, bubble_y2), 
            (cx, bubble_y2 + tail_h)
        ], fill=fill_color)
        
        # Finally, stamp pure black text inside container box
        text_x = bubble_x1 + pad_x
        text_y = bubble_y1 + pad_y - 2
        draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=self.font)
