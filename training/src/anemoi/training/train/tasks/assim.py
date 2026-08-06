# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import time 
import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.diagnostics.callbacks.plot_adapter import AssimilationPlotAdapter
from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class GraphAssim(BaseGraphModule):
    """Graph neural network assim for PyTorch Lightning."""

    task_type = "assim"
    
    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network assimilator. 

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        self._plot_adapter = AssimilationPlotAdapter(self)



    def _step(
        self,
        batch: dict[str, torch.Tensor],
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        x = {}

        # We agree to take several input if needed. 
        for dataset_name, dataset_batch in batch.items():
            LOGGER.info("SHAPES: batch.shape = %s, input_step = %d, output_step = %d", list(dataset_batch.shape), self.n_step_input, self.n_step_output )
            LOGGER.info("The last step should also be in the output.")
            assert (
            self.n_step_input == list(dataset_batch.shape)[1]
        ), f"The time dimension does not correspond. Your last input step should also be in your input. So {self.n_step_input } should be equal to {list(dataset_batch.shape)[1]}."
            x[dataset_name] = dataset_batch[:,:self.n_step_input][...,self.data_indices[dataset_name].data.input.full]

            if len(x[dataset_name].shape)<5: 
                LOGGER.info ("unsqueeze data input")
                x[dataset_name] = x[dataset_name].unsqueeze(2)
            LOGGER.info(f"Shape : {x[dataset_name].shape}  for {dataset_name}")


        y_pred = self(x)
        #print("SHAPE DE Y_PRED:",y_pred[dataset_name].shape)

        # We take the last input 
        y = {}
        for dataset_name, dataset_batch in batch.items():
            y[dataset_name] = dataset_batch[:,-1][...,self.data_indices[dataset_name].data.output.full]
            LOGGER.info(f"Shape output : {y[dataset_name].shape}  for {dataset_name}")
            if len(y[dataset_name].shape)<5: 
                LOGGER.info ("unsqueeze data output")
                y[dataset_name] = y[dataset_name].unsqueeze(2)

        #print("X shape: ", x["increments_input"].shape)
        #print("Y shape: ", y["increments_target"].shape)

        #mse = ((x["increments_input"]-y["increments_target"])**2).mean()
        #mse = torch.nn.MSELoss()(x["increments_input"],y["increments_target"])
        #print("MSE ENTRE X ET Y:",mse)
        #print("X ET Y ÉGAUX?", torch.equal(x["increments_input"],y["increments_target"]))

        # y includes the auxiliary variables, so we must leave those out when computing the loss
        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            rollout_step=0,
            training_mode=True,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        # All tasks return (loss, metrics, list of per-step dicts) for consistent plot callback contract.
        return loss, metrics, [y_pred]


    def on_train_epoch_start(self): 
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        duration = time.time() - self.epoch_start_time
        self.epoch_durations.append(duration)
        LOGGER.info(f"Epoch duration : {duration}")
        self.log("epoch_duration",duration,on_epoch=True, on_step=False, prog_bar=False, logger=self.logger_enabled,)

    def on_train_end(self):
        if self.epoch_durations:
            avg = sum(self.epoch_durations) / len(self.epoch_durations)
            LOGGER.info(f"Mean duration of an epoch : {avg:.2f}s")


