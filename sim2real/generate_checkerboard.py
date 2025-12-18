#!/usr/bin/env python3
"""
체커보드 패턴 PDF 생성기

Hand-eye calibration을 위한 프린트용 체커보드를 생성합니다.

Usage:
    python generate_checkerboard.py

Output:
    checkerboard_7x10_25mm.pdf - A4 용지에 프린트 가능한 체커보드
"""

import numpy as np

# PDF 생성을 위해 reportlab 사용
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
except ImportError:
    print("reportlab이 설치되어 있지 않습니다.")
    print("설치: pip install reportlab")
    exit(1)

# 체커보드 설정
ROWS = 10        # 세로 사각형 개수
COLS = 7         # 가로 사각형 개수
SQUARE_SIZE = 25  # 각 사각형 크기 (mm)

# 내부 코너 개수 (OpenCV에서 사용하는 값)
INNER_CORNERS = (COLS - 1, ROWS - 1)  # (6, 9)

OUTPUT_FILE = "/home/fhekwn549/doosan_ws/src/e0509_gripper_description/scripts/sim2real/checkerboard_7x10_25mm.pdf"


def generate_checkerboard_pdf():
    """A4 용지에 체커보드 PDF 생성"""

    # 페이지 크기 (A4: 210x297mm)
    page_width, page_height = A4

    # 체커보드 전체 크기
    board_width = COLS * SQUARE_SIZE * mm
    board_height = ROWS * SQUARE_SIZE * mm

    # 중앙 배치를 위한 시작 위치 계산
    start_x = (page_width - board_width) / 2
    start_y = (page_height - board_height) / 2

    # PDF 생성
    c = canvas.Canvas(OUTPUT_FILE, pagesize=A4)

    # 체커보드 그리기
    for row in range(ROWS):
        for col in range(COLS):
            x = start_x + col * SQUARE_SIZE * mm
            y = start_y + (ROWS - 1 - row) * SQUARE_SIZE * mm  # Y축 반전

            # 짝수/홀수에 따라 흰색/검은색
            if (row + col) % 2 == 0:
                c.setFillColorRGB(0, 0, 0)  # 검은색
            else:
                c.setFillColorRGB(1, 1, 1)  # 흰색

            c.rect(x, y, SQUARE_SIZE * mm, SQUARE_SIZE * mm, fill=1)

    # 테두리 (체커보드 경계)
    c.setStrokeColorRGB(0.5, 0.5, 0.5)
    c.setLineWidth(0.5)
    c.rect(start_x, start_y, board_width, board_height, fill=0)

    # 정보 텍스트 추가
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 10)

    text_y = start_y - 20
    c.drawString(start_x, text_y, f"Checkerboard: {COLS}x{ROWS} squares, {SQUARE_SIZE}mm each")
    c.drawString(start_x, text_y - 12, f"Inner corners (for OpenCV): {INNER_CORNERS[0]} x {INNER_CORNERS[1]}")
    c.drawString(start_x, text_y - 24, f"Total size: {COLS * SQUARE_SIZE}mm x {ROWS * SQUARE_SIZE}mm")

    # 방향 표시 (원점 마커)
    marker_x = start_x + SQUARE_SIZE * mm / 2
    marker_y = start_y + board_height - SQUARE_SIZE * mm / 2
    c.setFillColorRGB(1, 0, 0)
    c.circle(marker_x, marker_y, 3, fill=1)
    c.drawString(marker_x + 5, marker_y - 3, "Origin")

    c.save()

    print("=" * 60)
    print("체커보드 PDF 생성 완료!")
    print("=" * 60)
    print(f"\n파일: {OUTPUT_FILE}")
    print(f"\n체커보드 정보:")
    print(f"  - 사각형: {COLS}x{ROWS} ({COLS*ROWS}개)")
    print(f"  - 사각형 크기: {SQUARE_SIZE}mm")
    print(f"  - 내부 코너 (OpenCV): {INNER_CORNERS[0]} x {INNER_CORNERS[1]}")
    print(f"  - 전체 크기: {COLS * SQUARE_SIZE}mm x {ROWS * SQUARE_SIZE}mm")
    print(f"\n프린트 시 주의사항:")
    print("  1. '실제 크기'로 프린트하세요 (크기 조정 없음)")
    print("  2. 프린트 후 사각형 크기가 25mm인지 자로 확인하세요")
    print("  3. 평평한 판 (포맥스 등)에 붙여서 사용하세요")
    print("=" * 60)


if __name__ == "__main__":
    generate_checkerboard_pdf()
