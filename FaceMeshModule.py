import cv2
import mediapipe as mp
import time

GREEN = (0, 255, 0)


class FaceMeshDetector:

    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_con=0.5,
                 min_tracking_con=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_con = min_detection_con
        self.min_tracking_con = min_tracking_con

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks,
                                                 self.min_detection_con, self.min_tracking_con)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362, 476, 473, 474]

        self.RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]
        self.ALL_EYE_LANDMARKS = self.LEFT_EYE_LANDMARKS + self.RIGHT_EYE_LANDMARKS

        # coordiantes we need for EAR, for more info see this paper:
        # https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
        self.LEFT_IRIS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_IRIS = [33, 160, 158, 133, 153, 144]

        self.BOTH_IRISES = self.LEFT_IRIS + self.RIGHT_IRIS

    def findMeshFaces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(img_rgb)

        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)

                face = []
                for _, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, GREEN, 1)

                    face.append([x, y])
                faces.append(face)

        return img, faces

    def findMeshIrises(self, img, draw=True, color=GREEN):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(img_rgb)
        eye_landmarks = []
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                eye_landmarks = []
                face = []
                for i, lm in enumerate(faceLms.landmark):
                    if i in self.BOTH_IRISES:
                        h, w, ic = img.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        eye_landmarks.append((x, y))
                        # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, GREEN, 1)
                        if draw:
                            cv2.circle(img, (x, y), 1, color, -1)
                        face.append([x, y])

                faces.append(face)

        return img, faces, eye_landmarks


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceMeshDetector(refine_landmarks=True)
    while True:
        success, img = cap.read()
        img, faces = detector.findMeshFaces(img)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, GREEN, 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
