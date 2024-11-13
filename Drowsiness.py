import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
from collections import deque
from ultralytics import YOLO

def calculate_ear(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_head_pose(face_landmarks, frame_width, frame_height):
    # 머리 포즈 계산을 위한 주요 랜드마크
    nose = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    mouth_left = face_landmarks.landmark[57]
    mouth_right = face_landmarks.landmark[287]

    # 3D 좌표로 변환
    nose_3d = np.array([nose.x * frame_width, nose.y * frame_height, nose.z * 3000])
    left_eye_3d = np.array([left_eye.x * frame_width, left_eye.y * frame_height, left_eye.z * 3000])
    right_eye_3d = np.array([right_eye.x * frame_width, right_eye.y * frame_height, right_eye.z * 3000])

    # 머리 기울기 계산
    eye_center = (left_eye_3d + right_eye_3d) / 2
    vertical_angle = np.arctan2(nose_3d[1] - eye_center[1], nose_3d[2] - eye_center[2])
    return np.degrees(vertical_angle)

def detect_drowsiness(video_paths):
    # YOLO 모델 초기화
    yolo_model = YOLO('yolov8n.pt')

    # MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # 눈 랜드마크 인덱스
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    # 파라미터 설정
    EAR_THRESHOLD = 0.2
    HEAD_ANGLE_THRESHOLD = 20  # 머리 기울기 임계값
    MOVEMENT_THRESHOLD = 30    # 움직임 임계값
    CONSEC_FRAMES_THRESHOLD = 10
    
    # 이전 프레임의 머리 위치를 저장하는 큐
    head_positions = deque(maxlen=10)

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        prev_time = time.time()
        drowsy_counter = 0
        prev_head_angle = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            frame = cv2.resize(frame, (1280, 720))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO로 사람 검출
            yolo_results = yolo_model(frame, classes=[0], conf=0.3)
            
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size == 0:
                        continue

                    # Face Mesh 처리
                    face_mesh_results = face_mesh.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                    
                    if face_mesh_results.multi_face_landmarks:
                        for face_landmarks in face_mesh_results.multi_face_landmarks:
                            # 머리 포즈 계산
                            head_angle = calculate_head_pose(face_landmarks, person_roi.shape[1], person_roi.shape[0])
                            
                            # 현재 머리 위치 저장
                            nose_pos = face_landmarks.landmark[1]
                            current_pos = (nose_pos.x, nose_pos.y, nose_pos.z)
                            head_positions.append(current_pos)

                            # 머리 움직임 분석
                            movement = 0
                            if len(head_positions) > 1:
                                prev_pos = head_positions[-2]
                                movement = np.sqrt(
                                    (current_pos[0] - prev_pos[0])**2 +
                                    (current_pos[1] - prev_pos[1])**2 +
                                    (current_pos[2] - prev_pos[2])**2
                                ) * 1000

                            # 눈 EAR 계산
                            try:
                                left_eye = [(int(landmark.x * person_roi.shape[1]) + x1, 
                                           int(landmark.y * person_roi.shape[0]) + y1) 
                                          for landmark in [face_landmarks.landmark[i] for i in LEFT_EYE]]
                                right_eye = [(int(landmark.x * person_roi.shape[1]) + x1, 
                                            int(landmark.y * person_roi.shape[0]) + y1) 
                                           for landmark in [face_landmarks.landmark[i] for i in RIGHT_EYE]]
                                
                                left_ear = calculate_ear(np.array(left_eye))
                                right_ear = calculate_ear(np.array(right_eye))
                                ear = (left_ear + right_ear) / 2.0
                            except:
                                ear = 1.0  # 눈이 잘 보이지 않는 경우

                            # 졸음 상태 판단 및 색상 초기화
                            is_drowsy = False
                            status = "Alert"
                            color = (0, 255, 0)  # 기본 색상 (초록색)
                            
                            # 1. EAR 기반 졸음 감지
                            if ear < EAR_THRESHOLD:
                                is_drowsy = True
                            
                            # 2. 머리 기울기 기반 졸음 감지
                            if abs(head_angle) > HEAD_ANGLE_THRESHOLD:
                                is_drowsy = True
                            
                            # 3. 급격한 머리 움직임 감지
                            if movement > MOVEMENT_THRESHOLD:
                                is_drowsy = True

                            if is_drowsy:
                                drowsy_counter += 1
                                if drowsy_counter >= CONSEC_FRAMES_THRESHOLD:
                                    status = "Drowsy"
                                    color = (0, 0, 255)  # 빨간색
                            else:
                                drowsy_counter = max(0, drowsy_counter - 1)

                            # 상태 표시
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{status}", (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # 추가 정보 표시
                            info_y = 30
                            cv2.putText(frame, f"Head Angle: {head_angle:.1f}", (10, info_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Movement: {movement:.1f}", (10, info_y + 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            if ear != 1.0:
                                cv2.putText(frame, f"EAR: {ear:.2f}", (10, info_y + 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()

# 비디오 파일 경로 설정
video_paths = ["data/test1.mp4", "data/test2.mp4"]
detect_drowsiness(video_paths)
