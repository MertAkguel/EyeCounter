import cv2
import numpy as np
import time
from FaceMeshModule import FaceMeshDetector
import pygame

pygame.init()

# you can download it here: https://pixabay.com/sound-effects/click-124467/
click_sound = pygame.mixer.Sound(
    r"C:\Users\Kleve\Downloads\click-124467.mp3")

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
colour = GREEN


def get_distance(point1, point2):
    """
    calculate Euclidean distance between point1 and point2

    :param point1: (x, y, z) landmark of point1
    :param point2: (x, y, z) landmark of point2
    :return: Euclidean distance between point1 and point2
    """

    distance = []
    for p1, p2 in zip(point1, point2):
        distance.append((p2 - p1) ** 2)
    return np.sqrt(np.sum(distance))


def get_eye_aspect_ratio(p1, p2, p3, p4, p5, p6):
    distance_p2_6 = get_distance(p2, p6)
    distance_p3_5 = get_distance(p3, p5)
    distance_p1_4 = get_distance(p1, p4)

    ear = (distance_p2_6 + distance_p3_5) / (2 * distance_p1_4)

    return ear


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(refine_landmarks=True)

    left_eye_counter = 0
    right_eye_counter = 0

    left_eye_opened = True
    right_eye_opened = True

    while True:
        success, img = cap.read()
        h, w, _ = img.shape

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        try:
            img, faces, eye_landmarks = detector.findMeshIrises(img, draw=True, color=colour)
            left_p = eye_landmarks[6:]
            right_p = eye_landmarks[:6]

            left_eye_ear = get_eye_aspect_ratio(left_p[1], left_p[4], left_p[5],
                                                left_p[0], left_p[2], left_p[3])
            right_eye_ear = get_eye_aspect_ratio(right_p[0], right_p[5], right_p[4],
                                                 right_p[1], right_p[3], right_p[2])
            avg_ear = (left_eye_ear + right_eye_ear) / 2.0

            if left_eye_ear < 0.15 and left_eye_opened:
                left_eye_counter += 1
                left_eye_opened = False
                pygame.mixer.Sound.play(click_sound)

            if left_eye_ear > 0.15:
                left_eye_opened = True

            if right_eye_ear < 0.15 and right_eye_opened:
                right_eye_counter += 1
                right_eye_opened = False
            if right_eye_ear > 0.15:
                right_eye_opened = True

        except Exception:
            pass

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, colour, 1)
        cv2.putText(img, f'Left EyeCounter: {left_eye_counter}', (20, h // 2), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)
        cv2.putText(img, f'Right EyeCounter: {right_eye_counter}', (20, h // 2 + 50), cv2.FONT_HERSHEY_PLAIN, 1, colour,
                    1)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
