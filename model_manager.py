import os
import logging
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
import torch

logger = logging.getLogger(__name__)


class WhisperModelManager:
    """Управление загрузкой и кэшированием модели faster-whisper."""

    _instance: Optional["WhisperModelManager"] = None
    _model: Optional[WhisperModel] = None

    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = "float16",
        model_name: str = "Systran/faster-whisper-large-v2",
        download_root: Optional[str] = None
    ):
        self.device = device
        self.compute_type = compute_type
        self.model_name = model_name
        self.download_root = download_root or os.path.join(
            os.getcwd(), "app", "models", "faster-whisper-large-v2")
        self.required_files = ["model.bin", "tokenizer.json",
                               "config.json", "preprocessor_config.json"]

    def __new__(cls, *args, **kwargs):
        """Синглтон: только один экземпляр менеджера."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self) -> WhisperModel:
        """Ленивая загрузка модели."""
        if self._model is not None:
            return self._model

        logger.info(f"Загрузка модели faster-whisper на {self.device}...")
        self._model = self._load_model()
        return self._model

    def _load_model(self) -> WhisperModel:
        """Пытается загрузить локально → иначе скачивает с HF."""
        model_dir = Path(self.download_root)

        # 1. Прямая локальная модель
        if self._is_valid_model_dir(model_dir):
            return self._load_from_path(model_dir)

        # 2. Модель в кэше Hugging Face
        hf_cache_dir = model_dir / "models--Systran--faster-whisper-large-v2"
        snapshot_path = self._find_latest_snapshot(hf_cache_dir)
        if snapshot_path and self._is_valid_model_dir(snapshot_path):
            return self._load_from_path(snapshot_path)

        # 3. Загрузка с HF
        logger.info("Локальная модель не найдена. Загрузка с Hugging Face...")
        model_dir.mkdir(parents=True, exist_ok=True)

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN не найден — загрузка может быть медленной или ограничена.")

        try:
            model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(model_dir),
                local_files_only=False,
            )
            logger.info("Модель успешно загружена с Hugging Face.")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки с Hugging Face: {e}")
            raise RuntimeError(
                "Не удалось загрузить модель faster-whisper.") from e

    def _is_valid_model_dir(self, path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        missing = [f for f in self.required_files if not (path / f).is_file()]
        if missing:
            logger.info(f"Отсутствуют файлы в {path}: {missing}")
            return False
        logger.info(f"Валидная модель найдена: {path}")
        return True

    def _find_latest_snapshot(self, hf_cache_dir: Path) -> Optional[Path]:
        snapshots_dir = hf_cache_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        hashes = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not hashes:
            return None

        latest = max(hashes, key=lambda x: x.name)
        candidate = latest
        if self._is_valid_model_dir(candidate):
            return candidate
        logger.warning(f"Последний snapshot повреждён: {candidate}")
        return None

    def _load_from_path(self, path: Path) -> WhisperModel:
        try:
            model = WhisperModel(
                model_size_or_path=str(path),
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True
            )
            logger.info(f"Модель загружена локально из: {path}")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки из {path}: {e}")
            raise

    def cleanup(self):
        """Очистка модели и GPU (по желанию)."""
        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Модель выгружена из памяти.")
