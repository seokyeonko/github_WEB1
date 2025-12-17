import os
import wave
import contextlib
import pandas as pd

# 사용자에게 경로 입력 받기
directory = input("경로를 입력하세요: ")

# .wav 파일 경로 및 재생 시간 정보를 저장할 리스트
wav_files = []

# 하위 경로 포함하여 "01_mixing" 폴더 속 .wav 파일 찾기
for root, dirs, files in os.walk(directory):
    if "02_segment" in root:
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                # wave 모듈로 재생시간 계산
                try:
                    with contextlib.closing(wave.open(full_path, 'r')) as f:
                        frames = f.getnframes()          # 총 프레임 수
                        rate = f.getframerate()          # 초당 프레임 수 (샘플레이트)
                        duration = frames / float(rate)  # 재생시간(초)
                        wav_files.append({'filename': file, 'duration_sec': round(duration, 2)})
                except wave.Error:
                    # 비정상 파일일 경우 건너뜀
                    print(f"⚠️ {file}은(는) 올바른 WAV 파일이 아닙니다.")
                    continue

# 결과를 DataFrame으로 변환
df = pd.DataFrame(wav_files)

# 바탕화면에 CSV 저장
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "wav_files_info.csv")
df.to_csv(desktop_path, index=False, encoding='utf-8')

print(f"✅ {desktop_path} 에 파일이 저장되었습니다.")
