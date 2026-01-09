#!/usr/bin/env python3
"""
간단한 이미지 수집 스크립트 (YOLO 학습용)

HSV 감지 없이 순수하게 이미지만 저장합니다.
라벨링은 Roboflow 등 외부 도구로 진행합니다.

Usage:
    python collect_images_simple.py
    python collect_images_simple.py --output ./my_dataset --flip

조작:
    Space: 이미지 저장
    a: 자동 저장 토글 (1초 간격)
    q: 종료
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import os
import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="YOLO용 이미지 수집")
    parser.add_argument("--output", type=str, default="./pen_images",
                       help="출력 폴더")
    parser.add_argument("--flip", action="store_true", default=False,
                       help="이미지 상하좌우 반전")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="자동 저장 간격 (초)")
    args = parser.parse_args()

    # 출력 폴더 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RealSense 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("=" * 60)
    print("YOLO 학습용 이미지 수집")
    print("=" * 60)
    print(f"출력 폴더: {output_dir.absolute()}")
    print(f"Flip: {args.flip}")
    print("=" * 60)
    print("조작:")
    print("  Space: 이미지 저장")
    print("  a: 자동 저장 토글")
    print("  q: 종료")
    print("=" * 60)
    print("\n다양한 위치/각도/거리에서 펜을 촬영하세요!")
    print("- 가까이 / 멀리")
    print("- 손에 들고 / 테이블 위에")
    print("- 다양한 배경")
    print("=" * 60)

    saved_count = 0
    auto_save = False
    last_save_time = 0

    try:
        while True:
            # 프레임 가져오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Flip
            if args.flip:
                color_image = cv2.flip(color_image, -1)

            display = color_image.copy()

            # 상태 표시
            cv2.putText(display, f"Saved: {saved_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Auto: {'ON' if auto_save else 'OFF'}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if auto_save else (255, 255, 255), 2)
            cv2.putText(display, "Space: Save | a: Auto | q: Quit", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 중앙 가이드라인
            h, w = display.shape[:2]
            cv2.line(display, (w//2, 0), (w//2, h), (100, 100, 100), 1)
            cv2.line(display, (0, h//2), (w, h//2), (100, 100, 100), 1)

            cv2.imshow("Image Collector", display)

            # 자동 저장
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if auto_save and (current_time - last_save_time) >= args.interval:
                filename = f"pen_{saved_count+1:04d}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), color_image)
                saved_count += 1
                last_save_time = current_time
                print(f"[Auto] Saved: {filename}")

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # 수동 저장
                filename = f"pen_{saved_count+1:04d}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), color_image)
                saved_count += 1
                print(f"Saved: {filename}")
            elif key == ord('a'):
                auto_save = not auto_save
                last_save_time = current_time
                print(f"Auto save: {'ON' if auto_save else 'OFF'}")

    except KeyboardInterrupt:
        pass

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"수집 완료! 총 {saved_count}장")
    print(f"저장 위치: {output_dir.absolute()}")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. https://roboflow.com 에서 무료 계정 생성")
    print("2. 새 프로젝트 생성 (Object Detection)")
    print("3. 이미지 업로드")
    print("4. 펜에 바운딩 박스 라벨링 (클래스: 'pen')")
    print("5. YOLOv8 포맷으로 Export")
    print("6. 학습 진행")
    print("=" * 60)


if __name__ == "__main__":
    main()
