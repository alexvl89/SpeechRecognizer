import gc
import os
import logging
from pathlib import Path
from typing import Optional

import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment, effects
import mimetypes
from model_manager import WhisperModelManager


try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

AUDIO_SAVE_NORM = Path("audio_files/normalized")
AUDIO_SAVE_NORM.mkdir(parents=True, exist_ok=True)



class SpeechRecognizerFast:
    """Класс для распознавания речи с использованием faster-whisper и (опционально) суммаризации текста."""

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # compute_type = "int8" if torch.cuda.is_available(
    # ) else "float32"  # int8 для CUDA, float32 для CPU
    batch_size =  5 # Увеличен для faster-whisper, так как он более оптимизирован
    _model_cache = None
    _summarizer_cache = None

    # Инициализируем менеджер один раз
    _model_manager = WhisperModelManager(
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8"
    )


    @staticmethod
    def _log_devices():
        """Вывод информации об устройствах."""
        if torch.cuda.is_available():
            logger.info(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA не доступна. Используется CPU.")

    @staticmethod
    def preprocess_audio(input_path: Path, output_path: Path) -> Path:
        """Преобразует аудиофайл (ogg, mp3, wav, flac и т.п.) → wav, нормализует и добавляет тишину."""
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")

        logger.info(f"Обработка файла: {input_path}")

        # Определяем формат по расширению или MIME-типу
        ext = input_path.suffix.lower().replace('.', '')
        mime = mimetypes.guess_type(str(input_path))[0] or ""
        if not ext and "audio/" in mime:
            ext = mime.split("/")[-1]

        try:
            audio = AudioSegment.from_file(input_path, format=ext or None)
        except Exception as e:
            raise RuntimeError(
                f"Ошибка при декодировании аудио ({input_path}): {e}")

        # Преобразуем в моно 16кГц 16-бит
        audio = (
            audio.set_channels(1)
            .set_frame_rate(16000)
            .set_sample_width(2)
        )

        # Нормализация
        audio = effects.normalize(audio)

        # Добавляем 3 секунды тишины только если длительность > 2 секунд
        if len(audio) < 2000:  # длина в миллисекундах
            audio += AudioSegment.silent(duration=3000)

        # Создание каталога для выходного файла
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format="wav")

        logger.info(f"Сохранено нормализованное аудио: {output_path}")

        # Удаляем исходный файл 
        input_path.unlink(missing_ok=True)

        return output_path

    @classmethod
    def transcribe_audio(cls, input_path: str) -> str:
        """Распознаёт речь из аудиофайла и возвращает текст."""
        cls._log_devices()

        input_path = Path(input_path)
        wav_path = AUDIO_SAVE_NORM / f"{input_path.stem}.wav"
        cls.preprocess_audio(input_path, wav_path)

        try:
            # model = cls._get_model()
            # Получаем модель через менеджер
            model = cls._model_manager.get_model()
            # Транскрибация с faster-whisper
            segments, info = model.transcribe(
                str(wav_path),
                beam_size=5,
                language="ru",
                # batch_size=cls.batch_size
            )

            # Объединяем текст из сегментов
            text = " ".join(segment.text for segment in segments).strip()
            logger.info(f"Распознанный текст ({len(text)} символов)")

            return text

        finally:
            # Удаляем временный WAV-файл
            wav_path.unlink(missing_ok=True)
            # Выгружаем модель из памяти
            try:
                cls._model_manager.cleanup()
            except Exception as e:
                logger.warning(f"Ошибка при выгрузке модели: {e}")

            # Принудительная очистка Python- и CUDA-памяти
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def summarize_text(cls, text: str, max_length: int = 60) -> Optional[str]:
        """Создаёт краткий пересказ текста с помощью Transformers."""
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers не установлены, пересказ недоступен.")
            return None

        if cls._summarizer_cache is None:
            logger.info("Загрузка модели суммаризации...")
            cls._summarizer_cache = pipeline(
                "summarization", model="cointegrated/rut5-base-summarizer"
            )

        summarizer = cls._summarizer_cache
        summary = summarizer(text, max_length=max_length,
                             min_length=10, do_sample=False)
        return summary[0]["summary_text"].strip()
