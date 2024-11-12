import cv2
import dlib
import torch
from scipy.spatial import distance as dist
from imutils import face_utils
from ultralytics import YOLO
from threading import Thread
from playsound import playsound
import numpy as np

def play_alarm(path):
    playsound(path)

def detect_drowsiness(video_paths):
    # YOLOv11m 모델 초기화
    model = YOLO('yolo11m.pt')

    # CUDA 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)

    # 얼굴 랜드마크 예측기 초기화
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 3D 모델 포인트
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    ear_threshold = 0.25
    consec_frames = 15  # 연속 프레임 수 조정

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frame_counter = 0
        flag = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 크기 조정
            frame = cv2.resize(frame, (640, 360))

            # YOLO를 사용하여 사람 감지
            results = model(frame, device=device)
            detections = results[0].boxes  # 감지된 객체의 바운딩 박스

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                cls = int(box.cls[0])  # 클래스 ID
                if cls == 0:  # class 0은 'person'
                    roi_gray = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

                    # 얼굴 랜드마크 추출
                    rects = dlib.get_frontal_face_detector()(gray, 0)
                    for rect in rects:
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # 2D 이미지 포인트
                        image_points = np.array([
                            (shape[30][0], shape[30][1]),  # Nose tip
                            (shape[8][0], shape[8][1]),    # Chin
                            (shape[36][0], shape[36][1]),  # Left eye left corner
                            (shape[45][0], shape[45][1]),  # Right eye right corner
                            (shape[48][0], shape[48][1]),  # Left Mouth corner
                            (shape[54][0], shape[54][1])   # Right mouth corner
                        ], dtype='double')

                        # 카메라 내부 매개변수
                        size = frame.shape
                        focal_length = size[1]
                        center = (size[1]/2, size[0]/2)
                        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

                        dist_coeffs = np.zeros((4, 1))  # 렌즈 왜곡 없음 가정
                        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

                        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), 
                                                                         rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                        for p in image_points:
                            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                        p1 = (int(image_points[0][0]), int(image_points[0][1]))
                        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                        cv2.line(frame, p1, p2, (255, 0, 0), 2)

                        # 회전 벡터를 사용하여 Euler 각도 계산
                        rmat, jac = cv2.Rodrigues(rotation_vector)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))

                        if angles[1] < -15:
                            GAZE = "Looking: Left"
                            cv2.putText(frame, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
                        elif angles[1] > 15:
                            GAZE = "Looking: Right"
                            cv2.putText(frame, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
                        else:
                            GAZE = "Forward"
                            cv2.putText(frame, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)

                        cv2.putText(frame, "rotation: {:.2f}".format(y), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if y > 0.6 or y < -0.6:
                            frame_counter += 1

                            if frame_counter >= consec_frames:
                                if not flag:
                                    flag = True
                                    # 알람을 재생하는 새로운 스레드 시작
                                    t = Thread(target=play_alarm, args=('../alarm_trimmed.mp3',))
                                    t.deamon = True
                                    t.start()

                                cv2.putText(frame, "LOOK AT ROAD", (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        else:
                            frame_counter = 0
                            flag = False

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()

# 비디오 파일 경로 설정
video_paths = ["data/test1.mp4", "data/test2.mp4"]
detect_drowsiness(video_paths)
