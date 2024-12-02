from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from googletrans import Translator
import atexit
from datetime import datetime
import torch


class Backtranslator:
    def __init__(self, input_file, output_file, model_name, entities_to_mask):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_file = input_file
        self.output_file = output_file
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                                     device=0 if torch.cuda.is_available() else -1,ignore_labels=['0'])
        self.translator = Translator()
        self.entities_to_mask = entities_to_mask

    # 시작 시간, 끝나는 시간, 실행에 걸린 시간
    def start_end(self):
        # Log start time
        start_time = datetime.now()
        print(f"Execution started at: {start_time}")

        # Register atexit to log end time
        def log_end_time():
            end_time = datetime.now()
            print(f"Execution ended at: {end_time}")
            print(f"Total execution time: {end_time - start_time}")

        atexit.register(log_end_time)

    def mask_named_entities_by_phrase(self, text):
        entities = self.ner_pipeline(text)
        placeholders = {}
        # 어절 단위 분리
        words = text.split()
        masked_words = words.copy()

        masked_indices = set()
        special_postpositions = ['은','는','에게','이라고','을','를','이',
                                 '가','의','에','과','와','로','으로','도',
                                 '에서','만','이나','나','까지','부터','보다',
                                 '께','처럼','이라도','라도','으로서','로서']

        for idx, entity in enumerate(entities):
            if entity['entity'] in self.entities_to_mask:  # 마스킹할 엔터티 리스트에 있는 경우만 선택
                token_start, token_end = entity.get('start', 0), entity.get('end', 0)
                for i, word in enumerate(words):
                    if i in masked_indices:
                        continue
                    word_start = text.find(word)
                    word_end = word_start + len(word)
                    # 어절 안에 보호할 태그가 있다면
                    if word_start <= token_start < word_end:
                        # 1단계: 어절에 특수기호가 포함되어 있고 '은'이나 '는'이 끝에 온다면 특수기호와 은,는 제외하고 마스킹
                        if word[-1] in [',', '.', '!', '?'] and (word[-2:] == '은?' or word[-2:] == '는?'):
                            mask = f"<@_{idx}>"
                            masked_words[i] = mask + word[-2:]
                            placeholders[mask] = word[:-2]
                        # 2단계: 보호될 엔터티가 포함된 어절이 조사로 끝나지 않는다면 그대로 마스킹
                        elif not any(word.endswith(post) for post in special_postpositions):
                            mask = f"<@_{idx}>"
                            masked_words[i] = mask
                            placeholders[mask] = word
                        # 3단계: 보호될 엔터티가 포함된 어절이 조사로 끝난다면 조사 복사 후 마스킹
                        else:
                            for post in special_postpositions:
                                if word.endswith(post):
                                    mask = f"<@_{idx}>"
                                    masked_words[i] = mask + post
                                    placeholders[mask] = word
                                    break
                        masked_indices.add(i)
                        break

        masked_text = ' '.join(masked_words)
        return masked_text, placeholders

    def restore_placeholders(self, text, placeholders):
        # 마스킹 된 채로 모두 어절 단위로 분리
        words = text.split()
        restored_words = []

        for word in words:
            restored_word = word
            for placeholder, original_text in placeholders.items():
                if placeholder in word:
                    # 마스크가 들어있는 어절 전체를 placeholder로 교체
                    restored_word = original_text
                    break
            restored_words.append(restored_word)

        return ' '.join(restored_words)

    def process_text(self):
        with open(self.input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                masked_text, placeholders = self.mask_named_entities_by_phrase(line)

                translated_en = self.translator.translate(masked_text, src='ko', dest='ja').text

                back_translated_ko = self.translator.translate(translated_en, src='ja', dest='ko').text

                restored_text = self.restore_placeholders(back_translated_ko, placeholders)

                outfile.write(f"{restored_text}\n")

        print("Translation, correction, and post-editing complete. File saved.")


if __name__ == "__main__":
    input_file_path = '/home/danny/test/BT_augmentation/original_data/nia_model3.txt'
    output_file_path = '/home/danny/test/BT_augmentation/paraphrased_data/nia_model3_trans.txt'
    model_name = "monologg/koelectra-base-v3-naver-ner"
    # "Leo97/KoELECTRA-small-v3-modu-ner"
    entities_to_mask = ['ORG-B', 'PER-B', 'CVL-B']  # 마스킹할 엔터티 리스트 정의
    # ["B-AM", "I-AM", "B-LC", "I-LC", "B-OG", "I-OG", "B-PT", "I-PT", "B-TM", "I-TM", "B-TI"]

    processor = Backtranslator(input_file=input_file_path, output_file=output_file_path, model_name=model_name, entities_to_mask=entities_to_mask)
    processor.start_end()
    processor.process_text()
