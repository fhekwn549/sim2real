#!/usr/bin/env python3
"""
Roboflow API 펜 감지 테스트

API 속도 테스트용 스크립트입니다.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from inference_sdk import InferenceHTTPClient


def main():
    # Roboflow API 클라이언트
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="V9TmNCJWP2x2HONL1IIq"
    )

    # RealSense 초기화
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("=" * 50)
    print("Roboflow API 펜 감지 테스트")
    print("=" * 50)
    print("q: 종료")
    print("=" * 50)

    frame_count = 0
    total_api_time = 0

    try:
        while True:
            # 프레임 가져오기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()

            # API 호출 (매 10프레임마다 - 너무 빠르면 API 제한)
            if frame_count % 10 == 0:
                # 이미지를 임시 파일로 저장
                temp_path = "/tmp/temp_frame.jpg"
                cv2.imwrite(temp_path, color_image)

                # API 호출 시간 측정
                start_time = time.time()
                try:
                    result = client.run_workflow(
                        workspace_name="pen-detect-vakdp",
                        workflow_id="find-pens-2",
                        images={"image": temp_path},
                        use_cache=True
                    )
                    api_time = time.time() - start_time
                    total_api_time += api_time

                    # 결과 파싱
                    print(f"\n[API 응답] 소요시간: {api_time:.3f}초")
                    print(f"결과: {result}")

                    # 결과가 있으면 시각화
                    if result and len(result) > 0:
                        cv2.putText(display, f"DETECTED - API: {api_time:.2f}s",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, f"NO DETECTION - API: {api_time:.2f}s",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                except Exception as e:
                    print(f"[API 오류] {e}")
                    cv2.putText(display, f"API ERROR", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_count += 1

            # FPS 표시
            cv2.putText(display, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Roboflow API Test", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # 통계 출력
        api_calls = frame_count // 10
        if api_calls > 0:
            print("\n" + "=" * 50)
            print(f"총 API 호출: {api_calls}회")
            print(f"평균 API 응답시간: {total_api_time/api_calls:.3f}초")
            print("=" * 50)

            if total_api_time/api_calls > 0.5:
                print("\n⚠️  API가 너무 느립니다 (0.5초 이상)")
                print("로컬 YOLOv8 학습을 권장합니다.")
            else:
                print("\n✓ API 속도 양호!")


if __name__ == "__main__":
    main()
