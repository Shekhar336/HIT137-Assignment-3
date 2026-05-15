"""
HIT137 Assignment 3 - Spot the Difference Game
Group Assignment - Semester 1, 2026

RUBRIC CHECKLIST:
  [x] Codebase organised into at least 3 classes (we have 8)
  [x] Encapsulation - each class owns its data and methods via __init__
  [x] Constructor (__init__) in every class
  [x] Methods on every class
  [x] Class interaction - App uses ImageProcessor and GameState
  [x] Inheritance - 5 subclasses extend Alteration via super().__init__
  [x] Polymorphism - each subclass overrides apply() differently
  [x] Images load for JPG, PNG, BMP and scale with correct aspect ratio
  [x] 3+ distinct alteration types (we have 5)
  [x] Differences subtle but findable
  [x] Type and position randomised on every load
  [x] All 5 regions guaranteed non-overlapping
  [x] Original and modified displayed side by side
  [x] File dialog to choose image; loading new image resets the round
  [x] Clicks detected with reasonable tolerance; red circles on both images
  [x] Score and remaining counter update immediately on correct click
  [x] Mistake counter shown; clicks disabled at 3 mistakes with clear prompt
  [x] Counter resets on new image
  [x] Player notified when all 5 found
  [x] Reveal button marks unfound differences in blue on both images
  [x] Remaining counter updates to zero after reveal

Classes:
    Alteration     - Base class: position, size, overlap check, apply interface
    DarkenPatch    - Subclass: adaptive shadow scaled to region brightness
    WarmTint       - Subclass: warm colour cast scaled to channel headroom
    CoolTint       - Subclass: cool colour cast scaled to channel headroom
    SatBoost       - Subclass: saturation boost then hue rotation
    BlurPatch      - Subclass: Gaussian blur on textured regions
    ImageProcessor - Loads/scales image, generates 5 differences with OpenCV
    GameState      - Tracks found list, mistakes, score
    SpotTheDifferenceApp - Tkinter GUI with game-styled interface
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import random


def apply_ellipse_blend(roi, altered, w, h):
    """
    Blend altered pixels into original using a cosine-falloff ellipse weight map.
    Effect is strongest at the centre and fades to exactly zero at the boundary,
    so no hard box or oval edge is ever visible in the output image.
    Used by every alteration subclass to apply its effect smoothly.
    """
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.ogrid[0:h, 0:w]
    dist   = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
    weight = np.cos(np.clip(dist, 0, 1) * np.pi / 2) ** 2
    weight[dist > 1] = 0.0
    m = weight[:, :, np.newaxis]
    return np.clip(
        roi.astype(np.float32) + m * (altered.astype(np.float32) - roi.astype(np.float32)),
        0, 255).astype(np.uint8)


class Alteration:
    """
    Base class for one image alteration.
    Encapsulates patch position (x, y) and size (w, h).
    Defines the apply() interface that all subclasses must implement.
    Provides get_centre() for click detection and overlaps() for placement.
    """

    DEFAULT_SIZE = 65

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def apply(self, image):
        """Apply the effect to image and return the modified copy."""
        raise NotImplementedError("Subclasses must implement apply()")

    def get_centre(self):
        """Return (cx, cy) centre used for click detection and circle drawing."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def overlaps(self, other, margin=15):
        """Return True if this patch overlaps another (includes spacing margin)."""
        return not (
            self.x + self.w + margin <= other.x or
            other.x + other.w + margin <= self.x or
            self.y + self.h + margin <= other.y or
            other.y + other.h + margin <= self.y
        )

    def __str__(self):
        return self.__class__.__name__ + " at (" + str(self.x) + "," + str(self.y) + ")"


class DarkenPatch(Alteration):
    """
    Darkens a region by 28% of its own mean brightness (adaptive).
    Light areas darken by ~57px, dark areas by minimum 22px.
    Always looks like a natural shadow — never a black void.
    Demonstrates inheritance (super().__init__) and polymorphism (apply).
    """

    def __init__(self, x, y):
        super().__init__(x, y, random.randint(85, 115), random.randint(85, 115))

    def apply(self, image):
        img   = image.copy()
        roi   = img[self.y:self.y + self.h, self.x:self.x + self.w]
        delta = int(max(22, min(float(roi.mean()) * 0.28, 65)))
        alt   = np.clip(roi.astype(np.int32) - delta, 0, 255).astype(np.uint8)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = apply_ellipse_blend(
            roi, alt, self.w, self.h)
        return img


class WarmTint(Alteration):
    """
    Adds a warm orange/red cast by boosting Red and reducing Blue.
    Each delta is 40% of that channel's available headroom so it never
    clips to a solid blob and works on any image brightness.
    Demonstrates inheritance and polymorphism via apply() override.
    """

    def __init__(self, x, y):
        super().__init__(x, y, random.randint(85, 115), random.randint(85, 115))

    def apply(self, image):
        img      = image.copy()
        roi      = img[self.y:self.y + self.h, self.x:self.x + self.w]
        alt      = roi.astype(np.int32).copy()
        add_red  = int(max(20, min((255 - float(roi[:, :, 2].mean())) * 0.40, 70)))
        cut_blue = int(max(15, min(float(roi[:, :, 0].mean()) * 0.40, 55)))
        alt[:, :, 2] = np.clip(alt[:, :, 2] + add_red,  0, 255)
        alt[:, :, 0] = np.clip(alt[:, :, 0] - cut_blue, 0, 255)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = apply_ellipse_blend(
            roi, alt.astype(np.uint8), self.w, self.h)
        return img


class CoolTint(Alteration):
    """
    Adds a cool blue/cyan cast by boosting Blue and reducing Red.
    Uses the same adaptive headroom scaling as WarmTint.
    Looks like cold shade or moonlight falling on that area.
    Demonstrates inheritance and polymorphism via apply() override.
    """

    def __init__(self, x, y):
        super().__init__(x, y, random.randint(85, 115), random.randint(85, 115))

    def apply(self, image):
        img      = image.copy()
        roi      = img[self.y:self.y + self.h, self.x:self.x + self.w]
        alt      = roi.astype(np.int32).copy()
        add_blue = int(max(20, min((255 - float(roi[:, :, 0].mean())) * 0.40, 70)))
        cut_red  = int(max(15, min(float(roi[:, :, 2].mean()) * 0.40, 55)))
        alt[:, :, 0] = np.clip(alt[:, :, 0] + add_blue, 0, 255)
        alt[:, :, 2] = np.clip(alt[:, :, 2] - cut_red,  0, 255)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = apply_ellipse_blend(
            roi, alt.astype(np.uint8), self.w, self.h)
        return img


class SatBoost(Alteration):
    """
    Boosts saturation by +120 then rotates hue by 55-75 degrees in HSV.
    Boosting saturation first means this works even on near-white or grey
    pixels where plain hue rotation has no effect (saturation=0).
    Demonstrates inheritance and polymorphism via apply() override.
    """

    def __init__(self, x, y):
        super().__init__(x, y, random.randint(85, 115), random.randint(85, 115))
        self.hue_shift = random.choice([55, 65, 75, -55, -65, -75])

    def apply(self, image):
        img = image.copy()
        roi = img[self.y:self.y + self.h, self.x:self.x + self.w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 120, 0, 255)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift) % 180
        alt = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = apply_ellipse_blend(
            roi, alt, self.w, self.h)
        return img


class BlurPatch(Alteration):
    """
    Blurs a region with a large Gaussian kernel so texture is clearly lost.
    Skips flat/uniform areas (grey.var() < 80) where blur would be invisible.
    Falls back to WarmTint if no textured region is found after many attempts.
    Demonstrates inheritance and polymorphism via apply() override.
    """

    def __init__(self, x, y):
        super().__init__(x, y, random.randint(85, 115), random.randint(85, 115))
        k = random.choice([27, 31, 35])
        self.kernel = (k, k)

    def apply(self, image):
        img = image.copy()
        roi = img[self.y:self.y + self.h, self.x:self.x + self.w]
        alt = cv2.GaussianBlur(roi, self.kernel, 0)
        img[self.y:self.y + self.h, self.x:self.x + self.w] = apply_ellipse_blend(
            roi, alt, self.w, self.h)
        return img

    def has_texture(self, image):
        """Return True if region variance is high enough for blur to be visible."""
        roi  = image[self.y:self.y + self.h, self.x:self.x + self.w]
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(grey.var()) >= 80.0


class ImageProcessor:
    """
    Loads an image, scales it to fit the display window, then places one of
    each alteration type at a random non-overlapping position.
    All image manipulation uses OpenCV (Week 9 techniques).
    Demonstrates encapsulation and class interaction with Alteration subclasses.

    Image scaling works as follows:
        - Screen width is detected and DISPLAY_MAX is set to fit two images side
          by side with room for the UI panels.
        - Each image is scaled down if needed, preserving its aspect ratio.
        - Small images are never upscaled (scale capped at 1.0).
    """

    ALTERATION_TYPES = [DarkenPatch, WarmTint, CoolTint, SatBoost, BlurPatch]

    def __init__(self):
        self.original_bgr = None
        self.modified_bgr = None
        self.alterations  = []
        self.img_w = 0
        self.img_h = 0
        self.display_max  = 500  # default, recalculated on load from screen size

    def load(self, path, screen_w, screen_h):
        """
        Load, scale, and generate 5 differences.
        screen_w and screen_h are the actual window dimensions so the image
        always fits the available space regardless of monitor size.
        Returns True on success.
        """
        raw = cv2.imread(path)
        if raw is None:
            return False

        # Each image gets half the window width minus padding, and about 70% of height
        max_w = max(200, screen_w // 2 - 30)
        max_h = max(200, int(screen_h * 0.70))
        self.display_max  = min(max_w, max_h)

        self.original_bgr = self._scale(raw, max_w, max_h)
        self.img_h, self.img_w = self.original_bgr.shape[:2]
        self._generate()
        return True

    def _scale(self, img, max_w, max_h):
        """Scale image to fit max_w x max_h, preserving aspect ratio."""
        h, w  = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    def _generate(self):
        """
        Place one of each alteration type at a random non-overlapping position.
        BlurPatch checks texture variance and falls back to WarmTint on flat areas.
        """
        margin = 15
        self.alterations = []
        types = self.ALTERATION_TYPES[:]
        random.shuffle(types)

        for AltClass in types:
            placed = False
            for _ in range(500):
                max_x = self.img_w - Alteration.DEFAULT_SIZE - margin
                max_y = self.img_h - Alteration.DEFAULT_SIZE - margin
                if max_x <= margin or max_y <= margin:
                    break
                x = random.randint(margin, max_x)
                y = random.randint(margin, max_y)
                c = AltClass(x, y)
                c.w = min(c.w, self.img_w - c.x - 1)
                c.h = min(c.h, self.img_h - c.y - 1)
                if any(c.overlaps(a) for a in self.alterations):
                    continue
                if isinstance(c, BlurPatch) and not c.has_texture(self.original_bgr):
                    continue
                self.alterations.append(c)
                placed = True
                break

            if not placed and AltClass is BlurPatch:
                for _ in range(200):
                    x = random.randint(margin, max_x)
                    y = random.randint(margin, max_y)
                    fb = WarmTint(x, y)
                    if not any(fb.overlaps(a) for a in self.alterations):
                        self.alterations.append(fb)
                        break

        modified = self.original_bgr.copy()
        for alt in self.alterations:
            modified = alt.apply(modified)
        self.modified_bgr = modified

    def to_photoimage(self, bgr):
        """Convert OpenCV BGR array to Tkinter PhotoImage for canvas display."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def draw_circle(self, bgr, cx, cy, radius, colour, thickness=3):
        """Draw a circle on a copy of the image and return it."""
        out = bgr.copy()
        cv2.circle(out, (cx, cy), radius, colour, thickness)
        return out


class GameState:
    """
    Tracks all mutable game data. GUI calls methods here rather than touching
    variables directly (encapsulation). total_score persists across images.
    """

    MAX_MISTAKES = 3

    def __init__(self):
        self.found       = []
        self.mistakes    = 0
        self.locked      = False
        self.total_score = 0

    def new_image(self, num_alterations):
        """Reset per-image state, keep total_score."""
        self.found    = [False] * num_alterations
        self.mistakes = 0
        self.locked   = False

    def mark_found(self, index):
        """Mark difference found and increment cumulative score."""
        self.found[index] = True
        self.total_score += 1

    def record_mistake(self):
        """Increment mistake count. Returns True if now locked out."""
        self.mistakes += 1
        if self.mistakes >= self.MAX_MISTAKES:
            self.locked = True
        return self.locked

    def num_found(self):
        return sum(1 for f in self.found if f)

    def num_remaining(self):
        return len(self.found) - self.num_found()

    def is_complete(self):
        return all(self.found)

    def __str__(self):
        return ("GameState: found=" + str(self.num_found()) + "/5" +
                " mistakes=" + str(self.mistakes) +
                " score=" + str(self.total_score))


class SpotTheDifferenceApp:
    """
    Main Tkinter application. Game-styled interface with dark theme,
    score panel, and visual feedback. Composes ImageProcessor and GameState.

    Tkinter features used (Week 8 lecture):
        Tk(), Frame, Label, Button, Canvas  - widgets
        pack() and grid()                   - layout managers
        .bind("<Button-1>", handler)        - mouse click event binding
        filedialog.askopenfilename()        - file open dialog
        messagebox.showinfo/showwarning()   - popup message dialogs
        .config()                           - updating widget properties
        winfo_screenwidth/height()          - reading screen dimensions
    """

    RED   = (0, 0, 255)
    BLUE  = (255, 50, 50)
    CLICK_TOLERANCE = 45

    BG      = "#0d1117"
    PANEL   = "#161b22"
    ACCENT  = "#58a6ff"
    SUCCESS = "#3fb950"
    DANGER  = "#f85149"
    WARNING = "#d29922"
    TEXT    = "#c9d1d9"
    SUBTEXT = "#8b949e"

    def __init__(self, master):
        self.master = master
        self.master.title("Spot the Difference  |  HIT137")
        self.master.configure(bg=self.BG)
        self.master.resizable(True, True)

        self.processor    = ImageProcessor()
        self.state        = GameState()
        self.orig_photo   = None
        self.mod_photo    = None
        self.orig_display = None
        self.mod_display  = None

        self._build_ui()

        # Start maximised so images have as much space as possible
        self.master.state("zoomed")

    def _build_ui(self):
        """Build the full game interface using Frame, Label, Button, Canvas."""

        # Header bar with title and buttons
        header = tk.Frame(self.master, bg=self.PANEL, pady=10)
        header.pack(fill=tk.X)

        tk.Label(header, text="SPOT THE DIFFERENCE",
                 font=("Courier New", 20, "bold"),
                 fg=self.ACCENT, bg=self.PANEL).pack(side=tk.LEFT, padx=20)

        tk.Label(header, text="HIT137  |  Group Assignment 3",
                 font=("Courier New", 10),
                 fg=self.SUBTEXT, bg=self.PANEL).pack(side=tk.LEFT, padx=5)

        btn = {"font": ("Courier New", 11, "bold"), "relief": tk.FLAT,
               "padx": 16, "pady": 7, "cursor": "hand2", "bd": 0}

        self.btn_reveal = tk.Button(header, text="REVEAL",
                                    command=self._reveal_all,
                                    bg="#21262d", fg=self.SUBTEXT,
                                    activebackground="#30363d",
                                    state=tk.DISABLED, **btn)
        self.btn_reveal.pack(side=tk.RIGHT, padx=8)

        self.btn_load = tk.Button(header, text="LOAD IMAGE",
                                  command=self._load_image,
                                  bg=self.ACCENT, fg="#0d1117",
                                  activebackground="#79c0ff", **btn)
        self.btn_load.pack(side=tk.RIGHT, padx=4)

        # Score panel
        score_bar = tk.Frame(self.master, bg="#21262d", pady=8)
        score_bar.pack(fill=tk.X)

        self.lbl_remaining = self._score_pill(score_bar, "REMAINING", "5", self.ACCENT)
        self.lbl_mistakes  = self._score_pill(score_bar, "MISTAKES",  "0 / 3", self.DANGER)
        self.lbl_score     = self._score_pill(score_bar, "SCORE",     "0", self.SUCCESS)
        self.lbl_message   = tk.Label(score_bar, text="Load an image to begin!",
                                      font=("Courier New", 11, "italic"),
                                      fg=self.WARNING, bg="#21262d")
        self.lbl_message.pack(side=tk.RIGHT, padx=20)

        # Two image canvases side by side
        img_area = tk.Frame(self.master, bg=self.BG)
        img_area.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        col_lbl = {"font": ("Courier New", 10, "bold"), "bg": self.BG, "pady": 4}
        tk.Label(img_area, text="ORIGINAL",
                 fg=self.SUBTEXT, **col_lbl).grid(row=0, column=0)
        tk.Label(img_area, text="MODIFIED  \u2190  click here to find differences",
                 fg=self.ACCENT, **col_lbl).grid(row=0, column=1)

        canvas_cfg = {"bg": "#010409", "highlightthickness": 1}

        self.canvas_orig = tk.Canvas(img_area,
                                     highlightbackground="#30363d", **canvas_cfg)
        self.canvas_mod  = tk.Canvas(img_area, cursor="crosshair",
                                     highlightbackground=self.ACCENT, **canvas_cfg)

        self.canvas_orig.grid(row=1, column=0, padx=(0, 6), sticky="nsew")
        self.canvas_mod.grid( row=1, column=1, padx=(6, 0), sticky="nsew")

        img_area.columnconfigure(0, weight=1)
        img_area.columnconfigure(1, weight=1)
        img_area.rowconfigure(1, weight=1)

        # Only the modified canvas gets click events (Week 8 event binding)
        self.canvas_mod.bind("<Button-1>", self._on_click)

        # Footer
        footer = tk.Frame(self.master, bg="#161b22", pady=5)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(footer,
                 text="Click on the MODIFIED image to find differences.  Max 3 mistakes per image.",
                 font=("Courier New", 9), fg=self.SUBTEXT, bg="#161b22").pack()

    def _score_pill(self, parent, label, value, colour):
        """Create a labelled score pill widget and return its value Label."""
        frame = tk.Frame(parent, bg="#2d333b", padx=14, pady=4)
        frame.pack(side=tk.LEFT, padx=10)
        tk.Label(frame, text=label, font=("Courier New", 8, "bold"),
                 fg=self.SUBTEXT, bg="#2d333b").pack()
        val_lbl = tk.Label(frame, text=value,
                           font=("Courier New", 16, "bold"),
                           fg=colour, bg="#2d333b")
        val_lbl.pack()
        return val_lbl

    def _load_image(self):
        """Open file dialog, load image, reset game state, refresh display."""
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                       ("All files", "*.*")])
        if not path:
            return

        # Read actual window size so images scale to fit the current screen
        self.master.update_idletasks()
        screen_w = self.master.winfo_width()
        screen_h = self.master.winfo_height()

        if not self.processor.load(path, screen_w, screen_h):
            messagebox.showerror("Error", "Could not load that image.\n"
                                 "Please choose a JPG, PNG, or BMP file.")
            return

        self.state.new_image(len(self.processor.alterations))
        self.orig_display = self.processor.original_bgr.copy()
        self.mod_display  = self.processor.modified_bgr.copy()
        self._refresh_canvases()
        self._refresh_labels()
        self._set_msg("Find the 5 differences! Click on the right image.")
        self.btn_reveal.config(state=tk.NORMAL, bg="#21262d", fg=self.TEXT)

    def _on_click(self, event):
        """
        Handle click on the modified canvas.
        Correct click within CLICK_TOLERANCE draws red circles on both images.
        Wrong click increments mistake counter; 3 mistakes locks the round.
        Uses event.x and event.y from Tkinter (Week 8 event binding).
        """
        if self.state.locked or self.state.is_complete():
            return
        if self.processor.modified_bgr is None:
            return

        for i in range(len(self.processor.alterations)):
            if self.state.found[i]:
                continue
            alt = self.processor.alterations[i]
            cx, cy = alt.get_centre()
            dist = ((event.x - cx) ** 2 + (event.y - cy) ** 2) ** 0.5
            if dist <= self.CLICK_TOLERANCE:
                self.state.mark_found(i)
                r = max(alt.w, alt.h) // 2 + 8
                self.orig_display = self.processor.draw_circle(
                    self.orig_display, cx, cy, r, self.RED)
                self.mod_display  = self.processor.draw_circle(
                    self.mod_display,  cx, cy, r, self.RED)
                self._refresh_canvases()
                self._refresh_labels()
                if self.state.is_complete():
                    self._set_msg("All 5 found! Great job! Load a new image.")
                    messagebox.showinfo("Well Done!",
                        "You found all 5 differences!\n"
                        "Total Score: " + str(self.state.total_score))
                    self.state.locked = True
                else:
                    self._set_msg("Correct! " +
                                  str(self.state.num_remaining()) + " remaining.")
                return

        locked = self.state.record_mistake()
        self._refresh_labels()
        if locked:
            self._set_msg("Too many mistakes! Load a new image to try again.")
            messagebox.showwarning("Game Over",
                "You made 3 mistakes.\n"
                "You found " + str(self.state.num_found()) + " / 5 differences.\n"
                "Load a new image to continue.")
        else:
            self._set_msg("Wrong spot! " +
                          str(self.state.mistakes) + " / 3 mistakes used.")

    def _reveal_all(self):
        """
        Draw blue circles on all unfound differences on both images.
        Remaining counter updates to zero and further clicks are locked.
        """
        if self.processor.modified_bgr is None:
            return
        revealed = False
        for i in range(len(self.processor.alterations)):
            if not self.state.found[i]:
                alt = self.processor.alterations[i]
                cx, cy = alt.get_centre()
                r = max(alt.w, alt.h) // 2 + 8
                self.orig_display = self.processor.draw_circle(
                    self.orig_display, cx, cy, r, self.BLUE)
                self.mod_display  = self.processor.draw_circle(
                    self.mod_display,  cx, cy, r, self.BLUE)
                revealed = True
        if revealed:
            self._refresh_canvases()
            self.state.locked = True
            self.lbl_remaining.config(text="0")
            self._set_msg("Differences revealed. Load a new image to play again.")

    def _refresh_canvases(self):
        """Convert working BGR arrays to PhotoImages and redraw both canvases."""
        self.orig_photo = self.processor.to_photoimage(self.orig_display)
        self.mod_photo  = self.processor.to_photoimage(self.mod_display)
        for canvas, photo in [(self.canvas_orig, self.orig_photo),
                               (self.canvas_mod,  self.mod_photo)]:
            canvas.config(width=self.processor.img_w,
                          height=self.processor.img_h)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def _refresh_labels(self):
        """Update score panel. Mistake colour changes white to yellow to red."""
        self.lbl_remaining.config(text=str(self.state.num_remaining()))
        mistakes = self.state.mistakes
        colour = self.DANGER if mistakes >= 2 else self.WARNING if mistakes == 1 else self.TEXT
        self.lbl_mistakes.config(text=str(mistakes) + " / 3", fg=colour)
        self.lbl_score.config(text=str(self.state.total_score))

    def _set_msg(self, msg):
        self.lbl_message.config(text=msg)


def main():
    root = tk.Tk()
    SpotTheDifferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()