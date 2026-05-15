# Spot the Difference Game
HIT137 – Assignment 3  
Semester 1, 2026

## Project Overview

This project is a Spot the Difference game created using Python. The game allows the user to load an image and find five hidden differences between the original and modified images. The modified image is generated automatically using OpenCV image processing techniques.

The project was developed as part of Assignment 3 for HIT137 and demonstrates the use of:
- Object-Oriented Programming (OOP)
- Tkinter GUI development
- OpenCV image processing
- Event handling in Python


# Features

## Image Processing Features
- Supports JPG, PNG, JPEG, and BMP images
- Automatically scales images to fit the application window
- Generates 5 random non-overlapping differences
- Uses feather blending for smooth and natural-looking changes
- Randomises alteration types and positions each time a new image is loaded

## Difference Types
The game includes five different image alteration effects:

### DarkenPatch
Creates darker regions that look similar to natural shadows.

### WarmTint
Adds warm orange and red colour tones.

### CoolTint
Adds cool blue and cyan colour tones.

### SatBoost
Boosts saturation and changes colour hue.

### BlurPatch
Applies Gaussian blur to textured image regions.

---

# GUI Features

- Dark-themed game interface
- Original and modified images displayed side-by-side
- Mouse click detection on modified image
- Score tracking system
- Remaining differences counter
- Mistake counter with 3-strike lockout
- Reveal button to display remaining differences
- Visual feedback using coloured circles
- Popup messages for game events

---

# Object-Oriented Programming Concepts Used

## Encapsulation
Classes store and manage their own data and methods.

Examples:
- `GameState`
- `ImageProcessor`

## Inheritance
Different alteration classes inherit from the parent `Alteration` class.

Examples:
- `DarkenPatch`
- `WarmTint`
- `CoolTint`
- `SatBoost`
- `BlurPatch`

## Polymorphism
Each alteration class overrides the `apply()` method differently.

## Composition
The `SpotTheDifferenceApp` class combines:
- Image processing
- Game state management
- GUI interaction

---

# Technologies Used

- Python 3
- Tkinter
- OpenCV
- NumPy
- Pillow (PIL)

---

# Libraries Required

Install the required libraries before running the program:

```bash
pip install opencv-python pillow numpy

Tkinter is included with standard Python installations.

How to Run the Program
Download or clone the project files
Open the project folder
Run the Python file:
python main.py

(Replace main.py with the actual filename if needed.)

How to Play
Click the LOAD IMAGE button
Select an image from your computer
Compare the original and modified images
Click on the modified image to find differences
Correct selections are marked with red circles
The player loses after 3 incorrect clicks
Use the REVEAL button to display remaining differences
Project Structure
project_folder/
│
├──spot_the_difference.py
├── README.md
└── game_report.docx

OpenCV Techniques Used:
Gaussian Blur
HSV colour manipulation
Image resizing and scaling
BGR to RGB conversion
Circle drawing using OpenCV
Variance checking for texture detection
Feather blending for smooth transitions
Future Improvements


Possible future improvements include:

Difficulty levels
Timer system
Sound effects
Leaderboard system
Multiplayer mode
Additional alteration effects
