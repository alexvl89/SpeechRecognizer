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

    # @classmethod
    # def _get_model(cls):
    #     """Ленивая загрузка модели faster-whisper."""
    #     if cls._model_cache is not None:
    #         return cls._model_cache

    #     logger.info(f"Загрузка модели faster-whisper ({cls.device})...")
    #     # Относительный путь *в windows убрать первый /)
    #     model_path = "app2/models/faster-whisper-large-v2"
    #     # Абсолютный путь для отладки
    #     full_path = os.path.join(os.getcwd(), model_path)
    #     # Папка, где должна храниться модель
    #     # full_path = os.path.join(
    #     #     os.getcwd(), "app", "models", "faster-whisper-large-v2")
    #     logger.info(f"Проверка пути: {full_path}")

    #     if os.path.exists(full_path) and os.path.isdir(full_path):
    #         try:
    #             logger.info(f"Содержимое {full_path}: {os.listdir(full_path)}")

    #             # Шаг 1: Проверяем наличие основных файлов в корне (если модель распакована напрямую)
    #             required_files = ["model.bin", "tokenizer.json"]
    #             missing_files = [f for f in required_files if not os.path.exists(
    #                 os.path.join(full_path, f))]

    #             if not missing_files:
    #                 # Пытаемся загрузить локальную модель
    #                 cls._model_cache = WhisperModel(
    #                     model_size_or_path=full_path,
    #                     device=cls.device,
    #                     compute_type=cls.compute_type,
    #                     local_files_only=True
    #                 )
    #                 logger.info("Локальная модель успешно загружена.")
    #                 return cls._model_cache

    #             # Шаг 2: Проверяем структуру кэша Hugging Face (models--.../snapshots/<хеш>/)
    #             huggingface_dir = os.path.join(
    #                 full_path, "models--Systran--faster-whisper-large-v2")
    #             if os.path.exists(huggingface_dir):
    #                 snapshots_dir = os.path.join(huggingface_dir, "snapshots")

    #                 if os.path.exists(snapshots_dir):
    #                     # Получаем список хешей версий (подкаталогов в snapshots)
    #                     snapshot_hashes = [
    #                         d for d in os.listdir(snapshots_dir)
    #                         if os.path.isdir(os.path.join(snapshots_dir, d))
    #                     ]
    #                     if snapshot_hashes:
    #                         # Берём последний по алфавиту хеш (обычно самый свежий)
    #                         latest_hash = max(snapshot_hashes)
    #                         model_path = os.path.join(
    #                             snapshots_dir, latest_hash)

    #                         # Проверяем файлы внутри snapshot
    #                         missing_in_snapshot = [
    #                             f for f in required_files
    #                             if not os.path.exists(os.path.join(model_path, f))
    #                         ]
    #                         if not missing_in_snapshot:
    #                             logger.info(
    #                                 f"Модель найдена в snapshot: {model_path}. Загрузка...")
    #                             cls._model_cache = WhisperModel(
    #                                 model_size_or_path=model_path,
    #                                 device=cls.device,
    #                                 compute_type=cls.compute_type,
    #                                 local_files_only=True
    #                             )
    #                             logger.info(
    #                                 "Локальная модель из snapshot успешно загружена.")
    #                             return cls._model_cache
    #                         else:
    #                             logger.error(
    #                                 f"В snapshot {latest_hash} отсутствуют файлы: {missing_in_snapshot}")
    #                     else:
    #                         logger.error(
    #                             "Нет доступных snapshot-версий в папке snapshots/")
    #                 else:
    #                     logger.error(
    #                         "Папка snapshots/ не найдена в структуре Hugging Face.")
    #             else:
    #                 logger.warning(
    #                     "Структура Hugging Face (models--...) не обнаружена.")

    #         except Exception as e:
    #             logger.error(f"Ошибка загрузки локальной модели: {e}")

    #     try:
    #         # Если локальная модель отсутствует или не загрузилась → скачиваем
    #         logger.info("Загрузка модели с Hugging Face...")
    #         os.makedirs(full_path, exist_ok=True)
    #         hf_token = os.getenv("HF_TOKEN")
    #         if not hf_token:
    #             logger.warning(
    #                 "⚠️ Токен HF_TOKEN не найден в окружении — загрузка может быть ограничена.")

    #         cls._model_cache = WhisperModel(
    #             "large-v2",
    #             device=cls.device,
    #             compute_type=cls.compute_type,
    #             download_root=full_path,
    #             hf_token=hf_token
    #         )

    #         logger.info("✅ Модель успешно загружена с Hugging Face.")
    #         return cls._model_cache
    #     except Exception as e:
    #         logger.error(f"Ошибка загрузки модели с Hugging Face: {e}")


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

        # Нормализация и добавление 3 секунд тишины в конец
        audio = effects.normalize(audio)
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
            wav_path.unlink(missing_ok=True)
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
