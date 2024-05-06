from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class ResultSaver(Callback):
    def __init__(self, output="results") -> None:
        super().__init__()
        self._current_run = None
        self._current_dataset = None
        self._output_folder = Path(output)
        if not self._output_folder.exists():
            self._output_folder.mkdir(parents=True, exist_ok=True)

        self._current_results = {"preds": [], "targets": [], "images": []}
        self._current_dataloader_idx = 0

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._current_run = trainer.logger.experiment.name
        return super().on_test_start(trainer, pl_module)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if dataloader_idx != self._current_dataloader_idx:
            self.save_result()
            self._current_dataloader_idx = dataloader_idx
        return super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._current_dataset = trainer.test_dataloaders[dataloader_idx].dataset
        pred = outputs["pred"].detach().cpu().numpy()
        target = outputs["gt"].detach().cpu().numpy()
        indices = batch["index"].detach().cpu().numpy()
        filenames = self._current_dataset.filename(indices)
        self._current_results["preds"].extend(pred)
        self._current_results["targets"].extend(target)
        self._current_results["images"].extend(filenames)
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def save_result(self) -> None:
        print(f"Saving results {self._current_dataset.id}")

        output_file = self._output_folder / self._current_run / f"{self._current_dataset.id}.csv"
        if not output_file.parent.exists():
            output_file.parent.mkdir()
        for k, v in self._current_results.items():
            self._current_results[k] = np.hstack(v).ravel()

        pd.DataFrame(self._current_results).to_csv(output_file, index=False)
        self._current_results = {"preds": [], "targets": [], "images": []}

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_test_end(trainer, pl_module)
