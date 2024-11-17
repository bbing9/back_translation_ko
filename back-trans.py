import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from googletrans import Translator
import atexit
from datetime import datetime
import os
import torch


def start_end():
    # 실행 시작 시간 출력
    start_time = datetime.now()
    print(f"실행 시작: {start_time}")

    # 종료 시 실행 종료 시간 출력
    def log_end_time():
        end_time = datetime.now()
        print(f"실행 종료: {end_time}")
        print(f"실행에 걸린 시간: {end_time - start_time}")

    atexit.register(log_end_time)

start_end()

# MPS 디바이스 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# NER 모델 및 토크나이저 로드
model_name = "Leo97/KoELECTRA-small-v3-modu-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

# NER 파이프라인 설정
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device == "mps" else -1)

# 번역기 설정
translator = Translator()

# 보호할 엔티티 태그 리스트
protected_tags = {"B-AM"}


def mask_named_entities(text):
    """NER로 인식한 고유명사를 태깅 형태로 마스킹하여 보호"""
    entities = ner_pipeline(text)
    placeholders = {}
    masked_text = text

    # 고유명사에 태깅 이름 그대로 마스킹
    offset = 0  # 마스킹으로 인한 인덱스 변화를 추적하는 변수
    for entity in entities:
        if entity['entity'] in protected_tags:
            start, end = entity['start'] + offset, entity['end'] + offset
            original_text = text[entity['start']:entity['end']]

            # NER 태그명 그대로 마스킹 처리
            mask = f"<{entity['entity']}_{len(placeholders)}>"
            masked_text = masked_text[:start] + mask + masked_text[end:]

            # 오프셋을 업데이트하여 다음 인덱스 조정
            offset += len(mask) - (end - start)
            placeholders[mask] = original_text

    return masked_text, placeholders


def restore_placeholders(text, placeholders):
    """마스킹한 플레이스홀더를 고유명사로 복원"""
    for placeholder, original_text in placeholders.items():
        # 정확히 일치하는 placeholder를 원래 텍스트로 대체
        text = re.sub(re.escape(placeholder), original_text, text)
    return text


# 파일 경로 설정
input_file_path = os.path.expanduser("~/Desktop/J.Lee/PycharmProjects/pythonProject/test.txt")
output_file_path = os.path.expanduser("~/Desktop/J.Lee/PycharmProjects/pythonProject/bt_test.txt")

# txt 파일 처리
def read_txt_file(file_path):
    """txt 파일에서 텍스트 읽기"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()


def write_txt_file(file_path, lines):
    """txt 파일로 텍스트 저장"""
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


# txt 파일 읽기
lines = read_txt_file(input_file_path)

# 번역 및 파일 저장
translated_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    # 고유명사 마스킹
    masked_text, placeholders = mask_named_entities(line)

    # 1차 번역 (한국어 -> 영어)
    translated_en = translator.translate(masked_text, src="ko", dest="en").text

    # 2차 번역 (영어 -> 한국어)
    back_translated_ko = translator.translate(translated_en, src="en", dest="ko").text

    # 플레이스홀더 복원
    final_text = restore_placeholders(back_translated_ko, placeholders)

    # 번역 결과 저장
    translated_lines.append(final_text + "\n")

# 결과를 txt 파일로 저장
write_txt_file(output_file_path, translated_lines)

print("번역 및 파일 저장이 완료되었습니다.")