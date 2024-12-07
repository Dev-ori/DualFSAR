import os
import rarfile
import shutil
from pathlib import Path

def extract_rar_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.rar'):
                rar_path = os.path.join(root, file)
                # RAR 파일의 이름으로 새 디렉토리 만들기 (확장자 제외)
                dir_name = os.path.splitext(file)[0]
                extract_dir = os.path.join(root, dir_name)
                
                # 디렉토리가 없으면 생성
                if not os.path.exists(extract_dir):
                    os.makedirs(extract_dir)
                
                print(f"압축 해제 중: {file} -> {extract_dir}")
                
                try:
                    # RAR 파일 압축 해제
                    with rarfile.RarFile(rar_path) as rf:
                        # 각 파일에 대해 처리
                        for f in rf.namelist():
                            # 중첩된 디렉토리에서 파일 추출
                            if '/' in f:
                                # 파일 이름만 가져오기 (경로 제외)
                                extracted_filename = os.path.basename(f)
                                # 파일 데이터 읽기
                                data = rf.read(f)
                                # 새 경로에 파일 직접 쓰기
                                with open(os.path.join(extract_dir, extracted_filename), 'wb') as outfile:
                                    outfile.write(data)
                            else:
                                rf.extract(f, extract_dir)
                    print(f"성공: {file}")
                except Exception as e:
                    print(f"오류 발생 ({file}): {str(e)}")

# 실행할 디렉토리 경로 지정
base_directory = "/home/dhan/다운로드/FSAR_datasets/HMDB51/hmdb51_org"  # 여기에 실제 경로를 입력하세요
extract_rar_files(base_directory)
