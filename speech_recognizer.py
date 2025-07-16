from pydub import effects
import whisperx
import gc
import torch
import os
from pathlib import Path
# from main import start_bot
from pydub import AudioSegment

AUDIO_SAVE_NORM = "audio_files\normalized"


class SpeechRecognizer:


    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8"  # или "float16"/"float32"
    batch_size = 5
    hf_token = os.getenv('YOUR_HF_TOKEN')

    @staticmethod
    def log_devices():
        if torch.cuda.is_available():
            print("CUDA is available!")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is NOT available. Using CPU.")

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print("XPU is available")
        else:
            print("XPU is not available")

    @staticmethod
    def preprocess_audio(input_path: str, output_path: str) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Файл не найден: {input_path}")
        print("Файл найден:", input_path)

        # Преобразование в WAV 16bit mono 16kHz PCM + нормализация
        audio = AudioSegment.from_file(input_path, format="ogg")
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio = effects.normalize(audio)

        # Добавим тишину на случай коротких аудио
        silence = AudioSegment.silent(duration=5000)
        audio += silence

        audio.export(output_path, format="wav")
        print(f"Audio preprocessed and saved to: {output_path}")
        return output_path

    @classmethod
    def transcribe_audio(cls, input_ogg_path: str) -> str:
        cls.log_devices()
        wav_path = os.path.join(
            AUDIO_SAVE_NORM, os.path.basename(input_ogg_path))
        cls.preprocess_audio(input_ogg_path, wav_path)

        # 1. Распознавание речи
        model = whisperx.load_model(
            "large-v2", cls.device, compute_type=cls.compute_type)
        audio_tensor = whisperx.load_audio(wav_path)
        result = model.transcribe(
            audio_tensor, batch_size=cls.batch_size, language='ru')
        print("Распознанные сегменты (до alignment):")
        print(result["segments"])

        # # 2. Алигнмент
        # model_a, metadata = whisperx.load_align_model(
        #     language_code=result["language"],
        #     device=cls.device
        # )
        # result = whisperx.align(
        #     result["segments"], model_a, metadata, wav_path,
        #     device=cls.device,
        #     return_char_alignments=False
        # )
        # print("Сегменты после alignment:")
        # print(result["segments"])

        # # 3. Диаризация
        # diarize_model = whisperx.diarize.DiarizationPipeline(
        #     use_auth_token=cls.hf_token,
        #     device=cls.device
        # )
        # diarize_segments = diarize_model(wav_path)
        # result = whisperx.assign_word_speakers(diarize_segments, result)

        # print("Сегменты с указанием speaker ID:")
        # print(result["segments"])

        # Очистка
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Собираем итоговый текст
        text = " ".join([seg["text"] for seg in result["segments"]])
        return text.strip()

# # Получение токена из переменной окружения
# YOUR_HF_TOKEN = os.getenv('YOUR_HF_TOKEN')

# if torch.cuda.is_available():
#     print("CUDA is available!")
#     print(f"Device count: {torch.cuda.device_count()}")
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("CUDA is NOT available. Using CPU.")


# if torch.xpu.is_available():
#     print("xpu is available")
# else:
#     print("xpu is not available")

# device = "cpu"
# audio_file = "audio_2025-07-11_14-50-05.ogg"
# input_ogg = "audio_files/854924596_25.ogg"
# batch_size = 16 # reduce if low on GPU mem
# compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# # Проверка наличия файла
# if not os.path.exists(input_ogg):
#     raise FileNotFoundError(f"Файл не найден: {input_ogg}")
# print("Файл найден:", input_ogg)


# # Конвертация ogg → wav (моно, 16кГц, 16bit PCM)
# audio = AudioSegment.from_file(input_ogg, format="ogg")
# audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
# # Нормализация аудио
# audio = effects.normalize(audio)


# silence = AudioSegment.silent(duration=5000)  # 1 сек
# audio = audio + silence

# print(f"Duration (ms): {len(audio)}")  # минимум желательно > 1000

# input_ogg = "silence.wav"
# # Сохранить как WAV
# audio.export(input_ogg, format="wav")


# # # Можно "base" или "tiny"
# # model = whisperx.load_model("small", device, compute_type=compute_type)
# # # model = whisperx.load_model("large-v2", device, compute_type=compute_type)
# # result = model.transcribe(input_ogg, batch_size=5,
# #                           language='ru')
# # # result = model.transcribe(input_ogg, language='ru')
# # print(result)

# # normalized_wav = "normalized_audio.wav"
# # audio.export(normalized_wav, format="wav")

# # 1. Распознавание речи
# model = whisperx.load_model("large-v2", device, compute_type=compute_type)
# audio_tensor = whisperx.load_audio(input_ogg)
# result = model.transcribe(audio_tensor, batch_size=batch_size, language='ru')
# print(result["segments"])  # before alignment

# # 2. Алигнмент
# model_a, metadata = whisperx.load_align_model(
#     language_code=result["language"], device=device)
# result = whisperx.align(
#     result["segments"],
#     model_a,
#     metadata,
#     input_ogg,  # передаём путь, а не массив
#     device,
#     return_char_alignments=False
# )
# print(result["segments"])  # after alignment

# # 3. Диаризация
# diarize_model = whisperx.diarize.DiarizationPipeline(
#     use_auth_token=YOUR_HF_TOKEN,
#     device=device
# )

# diarize_segments = diarize_model(input_ogg)  # путь к wav-файлу
# result = whisperx.assign_word_speakers(diarize_segments, result)

# print(diarize_segments)
# print(result["segments"])  # с указанием speaker ID

# # Очистка GPU (если надо)
# gc.collect()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# # # Запуск Telegram-бота
# # if __name__ == "__main__":
# #     start_bot()

# print("Завершение цикла")
