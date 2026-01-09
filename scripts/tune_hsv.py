#!/usr/bin/env python3
"""
HSV 색상 튜닝 도구

슬라이더로 HSV 값을 조정하면서 펜 인식 결과를 실시간으로 확인
"""

import cv2
import numpy as np
import pyrealsense2 as rs

# 기본 HSV 값 (현재 설정)
hsv_values = {
    'h_low': 100,
    'h_high': 130,
    's_low': 100,
    's_high': 255,
    'v_low': 50,
    'v_high': 255
}

def nothing(x):
    pass

def main():
    # RealSense 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 윈도우 생성
    cv2.namedWindow('HSV Tuning')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Result')

    # 슬라이더 생성
    cv2.createTrackbar('H Low', 'HSV Tuning', hsv_values['h_low'], 179, nothing)
    cv2.createTrackbar('H High', 'HSV Tuning', hsv_values['h_high'], 179, nothing)
    cv2.createTrackbar('S Low', 'HSV Tuning', hsv_values['s_low'], 255, nothing)
    cv2.createTrackbar('S High', 'HSV Tuning', hsv_values['s_high'], 255, nothing)
    cv2.createTrackbar('V Low', 'HSV Tuning', hsv_values['v_low'], 255, nothing)
    cv2.createTrackbar('V High', 'HSV Tuning', hsv_values['v_high'], 255, nothing)
    cv2.createTrackbar('Flip', 'HSV Tuning', 0, 1, nothing)  # 0=off, 1=on

    print("=" * 50)
    print("HSV 튜닝 도구")
    print("=" * 50)
    print("슬라이더로 HSV 값을 조정하세요")
    print("파란색 펜 캡이 흰색 마스크에 나타나면 OK")
    print("q: 종료 및 값 출력")
    print("=" * 50)

    try:
        while True:
            # 프레임 가져오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Flip 적용
            flip = cv2.getTrackbarPos('Flip', 'HSV Tuning')
            if flip:
                color_image = cv2.flip(color_image, -1)

            # 슬라이더 값 읽기
            h_low = cv2.getTrackbarPos('H Low', 'HSV Tuning')
            h_high = cv2.getTrackbarPos('H High', 'HSV Tuning')
            s_low = cv2.getTrackbarPos('S Low', 'HSV Tuning')
            s_high = cv2.getTrackbarPos('S High', 'HSV Tuning')
            v_low = cv2.getTrackbarPos('V Low', 'HSV Tuning')
            v_high = cv2.getTrackbarPos('V High', 'HSV Tuning')

            # HSV 변환 및 마스크 생성
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower = np.array([h_low, s_low, v_low])
            upper = np.array([h_high, s_high, v_high])
            mask = cv2.inRange(hsv, lower, upper)

            # 마스크 적용 결과
            result = cv2.bitwise_and(color_image, color_image, mask=mask)

            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 가장 큰 컨투어 표시
            display = color_image.copy()
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > 50:
                    cv2.drawContours(display, [largest], -1, (0, 255, 0), 2)
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(display, f"Area: {area:.0f}", (cx+10, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # HSV 값 표시
            cv2.putText(display, f"H: {h_low}-{h_high}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"S: {s_low}-{s_high}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"V: {v_low}-{v_high}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 화면 표시
            cv2.imshow('HSV Tuning', display)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # 최종 값 출력
        print("\n" + "=" * 50)
        print("최종 HSV 값:")
        print("=" * 50)
        print(f"hue_low: int = {h_low}")
        print(f"hue_high: int = {h_high}")
        print(f"sat_low: int = {s_low}")
        print(f"sat_high: int = {s_high}")
        print(f"val_low: int = {v_low}")
        print(f"val_high: int = {v_high}")
        print("=" * 50)
        print("\n이 값을 pen_detector_advanced.py의")
        print("AdvancedDetectionConfig에 복사하세요.")


if __name__ == "__main__":
    main()
