import cv2
import mediapipe as mp

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

        # coordiantes we need for EAR, for more info see this paper:
        # https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
        self.LEFT_IRIS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_IRIS = [33, 160, 158, 133, 153, 144]

        self.BOTH_IRISES = self.LEFT_IRIS + self.RIGHT_IRIS

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
