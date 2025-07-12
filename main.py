from pydub import effects
import whisperx
import gc
import torch
import os
from pathlib import Path
from bot import start_bot
from pydub import AudioSegment


# Получение токена из переменной окружения
YOUR_HF_TOKEN = os.getenv('YOUR_HF_TOKEN')

if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is NOT available. Using CPU.")


if torch.xpu.is_available():
    print("xpu is available")
else:
    print("xpu is not available")

device = "cpu"
audio_file = "audio_2025-07-11_14-50-05.ogg"
input_ogg = "audio_files/854924596_25.ogg"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# Проверка наличия файла
if not os.path.exists(input_ogg):
    raise FileNotFoundError(f"Файл не найден: {input_ogg}")
print("Файл найден:", input_ogg)


# Конвертация ogg → wav (моно, 16кГц, 16bit PCM)
audio = AudioSegment.from_file(input_ogg, format="ogg")
audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
# Нормализация аудио
audio = effects.normalize(audio)
print(f"Duration (ms): {len(audio)}")  # минимум желательно > 1000

normalized_wav = "normalized_audio.wav"
audio.export(normalized_wav, format="wav")

# 1. Распознавание речи
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio_tensor = whisperx.load_audio(normalized_wav)
result = model.transcribe(audio_tensor, batch_size=batch_size, language='ru')
print(result["segments"])  # before alignment

# # 2. Алигнмент
# model_a, metadata = whisperx.load_align_model(
#     language_code=result["language"], device=device)
# result = whisperx.align(
#     result["segments"],
#     model_a,
#     metadata,
#     normalized_wav,  # передаём путь, а не массив
#     device,
#     return_char_alignments=False
# )
# print(result["segments"])  # after alignment

# # 3. Диаризация
# diarize_model = whisperx.diarize.DiarizationPipeline(
#     use_auth_token=YOUR_HF_TOKEN,
#     device=device
# )

# diarize_segments = diarize_model(normalized_wav)  # путь к wav-файлу
# result = whisperx.assign_word_speakers(diarize_segments, result)

# print(diarize_segments)
# print(result["segments"])  # с указанием speaker ID

# Очистка GPU (если надо)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# # Запуск Telegram-бота
# if __name__ == "__main__":
#     start_bot()

print("Завершение цикла")
