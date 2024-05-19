from django.shortcuts import render
from django.http import JsonResponse
from .main import GazeTrackingSession
from .models import GazeTrackingResult
import cv2
import pandas as pd
import shutil
import base64
import io
from PIL import Image
import numpy as np

# 전역 변수 선언
gaze_session = GazeTrackingSession()

def start_gaze_tracking_view(request):
    gaze_session.start_eye_tracking()  # 시선 추적 시작
    return JsonResponse({"message": "Gaze tracking started"}, status=200)

def apply_gradient(center, inner_radius, outer_radius, color, image):
    overlay = image.copy()
    cv2.circle(overlay, center, inner_radius, (255, 255, 255), -1)  # 안쪽 원 (흰색)
    cv2.circle(overlay, center, outer_radius, color, -1)  # 바깥쪽 원 (그라데이션)
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

def draw_heatmap(image, section_counts):
    if image is not None:
        height, width, _ = image.shape
        section_centers = {
            "A": (int(width / 6), int(height / 4)),
            "B": (int(width / 2), int(height / 4)),
            "C": (int(5 * width / 6), int(height / 4)),
            "D": (int(width / 6), int(3 * height / 4)),
            "E": (int(width / 2), int(3 * height / 4)),
            "F": (int(5 * width / 6), int(3 * height / 4))
        }
        max_count = max(section_counts.values(), default=1)  # 최대 값이 0일 경우 1로 설정하여 ZeroDivisionError 방지
        if max_count == 0:
            max_count = 1  # 최대 값이 0일 경우 1로 설정
        for section, count in section_counts.items():
            if section in section_centers:
                center = section_centers[section]
                percent = int(100 * (count / max_count))  
                for i in range(1, 5):  
                    inner_radius = 30 * i
                    outer_radius = 30 * i + percent  
                    if i == 1:
                        color = (0, 0, 255)  
                    elif i == 2:
                        color = (0, 200, 200)  
                    elif i == 3:
                        color = (0, 200, 0)  
                    elif i == 4:
                        color = (200, 0, 0)  
                    apply_gradient(center, inner_radius, outer_radius, color, image)

def stop_gaze_tracking_view(request):
    csv_filename = gaze_session.stop_eye_tracking()  # 섹션 및 횟수를 저장하고 시선 추적 종료
    section_data = pd.read_csv(csv_filename)
    section_counts = dict(zip(section_data["Section"], section_data["Count"]))

    image_path = "C:/KJE/IME_graduation_AI/Back_AI_connect-main/Eyetrack/0518/image.png"
    original_image = cv2.imread(image_path)  # 이미지 로드

    if original_image is None:
        return JsonResponse({"message": "Image not found"}, status=404)

    # heatmap 그리기
    heatmap_image = original_image.copy()
    draw_heatmap(heatmap_image, section_counts)

    # 이미지를 base64로 인코딩하여 문자열로 변환
    _, buffer = cv2.imencode('.png', heatmap_image)
    encoded_image_string = base64.b64encode(buffer).decode('utf-8')

    # # 이미지 저장
    # heatmap_image_filename = "C:/KJE/IME_graduation_AI/Back_AI_connect-main/Eyetrack/0518/heatmap_result.png"
    # cv2.imwrite(heatmap_image_filename, heatmap_image)

    # GazeTrackingResult 모델에 이미지 데이터 저장
    gaze_tracking_result = GazeTrackingResult.objects.create(encoded_image=encoded_image_string)

    return JsonResponse({"message": "Gaze tracking stopped", "image_data": gaze_tracking_result.encoded_image}, status=200)

