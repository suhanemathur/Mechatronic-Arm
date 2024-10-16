Clamp between -1 and 1
        angle = math.degrees(math.acos(cosine_angle))
        return angle
    except ZeroDivisionError:
        # Handle cases where the points are too close, or collinear
        return 0.0


# Function to check if the hand is in a fist
def is_fist(landmarks):
    finger_tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Thumb tip, Index tip, etc.
    palm_base = landmarks[0]  # Wrist landmark
    fist_threshold = 40  # Threshold for fist dete