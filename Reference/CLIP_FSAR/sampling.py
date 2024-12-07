import cv2
import os
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_dir, output_base_dir='Sampling'):
    # 모든 클래스 디렉토리 순회
    for class_name in os.listdir(video_dir):
        class_path = os.path.join(video_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # 각 비디오 파일 처리
        video_files = [f for f in os.listdir(class_path) if f.endswith(('.avi', '.mp4'))]
        for video_file in tqdm(video_files, desc=f"처리 중인 클래스: {class_name}"):
            video_path = os.path.join(class_path, video_file)
            
            # 출력 디렉토리 생성
            output_dir = os.path.join(output_base_dir, class_name, video_file)
            os.makedirs(output_dir, exist_ok=True)

            # 비디오 열기
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            # 프레임 추출 진행바
            with tqdm(total=total_frames, desc=f"프레임 추출: {video_file}", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 프레임 저장 (3자리 숫자로 번호 부여)
                    frame_name = f"frame_{frame_count:03d}.jpg"
                    output_path = os.path.join(output_dir, frame_name)
                    cv2.imwrite(output_path, frame)
                    frame_count += 1
                    pbar.update(1)

            cap.release()
            print(f"완료: {video_file} - {frame_count} 프레임 추출됨")

if __name__ == "__main__":
    video_directory = "/home/dhan/다운로드/FSAR_datasets/HMDB51/hmdb51_org"
    extract_frames(video_directory)
