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
        obs_zeros = None

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

            if validation_mode: 
                if dataset_name == "innovations": 
                    innovations = self.model.post_processors[dataset_name](x[dataset_name],in_place=False)
                    obs_zeros = innovations == 0
                    print("POINTS D'OBS TRAIN:", torch.count_nonzero(~obs_zeros)[0])

                if dataset_name == "observations": 
                    obs = self.model.post_processors[dataset_name](x[dataset_name],in_place=False)
                    obs_zeros = torch.isnan(obs)
                    print("POINTS D'OBS TRAIN:", torch.count_nonzero((~obs_zeros)[0]))
                
        y_pred = self(x)

        # We take the last input 
        y = {}
        y_metric = {}

        y_pred_metric = {}
        zeros_metric = {}
        ones_metric ={} 
        mask_metric = {}
        mask_tilde_metric = {}
        

        for dataset_name, dataset_batch in batch.items():
            print("Y dataset name:", dataset_name)
            y[dataset_name] = dataset_batch[:,-1][...,self.data_indices[dataset_name].data.output.full]
            LOGGER.info(f"Shape output : {y[dataset_name].shape}  for {dataset_name}")
            if len(y[dataset_name].shape)<5: 
                LOGGER.info ("unsqueeze data output")
                y[dataset_name] = y[dataset_name].unsqueeze(2)
            if validation_mode: 
                y_metric[dataset_name] = y[dataset_name].clone().detach()
                y_pred_metric[dataset_name] = y_pred[dataset_name].clone().detach()
                mask_metric[dataset_name] = obs_zeros.clone().long()
                mask_tilde_metric[dataset_name] = (~obs_zeros).long()
                print("NB NON OBS:", torch.count_nonzero(mask_metric[dataset_name][0]))
                print("NB OBS:", torch.count_nonzero(mask_tilde_metric[dataset_name][0]))
                print("SUM TOTAL", torch.count_nonzero(mask_metric[dataset_name][0]) + torch.count_nonzero(mask_tilde_metric[dataset_name][0]))
                zeros_metric[dataset_name] = torch.zeros(obs_zeros.shape).to(mask_metric[dataset_name].device)
                ones_metric[dataset_name] = torch.ones(obs_zeros.shape).to(mask_metric[dataset_name].device)

        # y includes the auxiliary variables, so we must leave those out when computing the loss


        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            step=0,
            training_mode=True,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        if validation_mode and obs_zeros is not None: 
            for dataset_name, dataset_batch in batch.items():
                if len(self.data_indices[dataset_name].data.output.full) != 0: #si le dataset est en output, on applique le masque: 

                    y_metric[dataset_name][obs_zeros] = 0
                    y_pred_metric[dataset_name][obs_zeros] = 0

                    mask_tilde_metric_full, zeros_metric_full, grid_shard_slice = self._prepare_tensors_for_loss(
                    mask_tilde_metric[dataset_name],
                    zeros_metric[dataset_name],
                    validation_mode=validation_mode,
                    dataset_name=dataset_name,
                    )

                    _, ones_metric_full, _ = self._prepare_tensors_for_loss(
                    mask_tilde_metric[dataset_name],
                    ones_metric[dataset_name],
                    validation_mode=validation_mode,
                    dataset_name=dataset_name,
                    )

                    W_norm = self._compute_loss(
                    y_pred=ones_metric_full,
                    y=zeros_metric_full,
                    grid_shard_slice=grid_shard_slice,
                    dataset_name=dataset_name,
                    )

                    wo_norm = self._compute_loss(
                    y_pred=mask_tilde_metric_full,
                    y=zeros_metric_full,
                    grid_shard_slice=grid_shard_slice,
                    dataset_name=dataset_name,
                    )

                    mask_metric_full, _, grid_shard_slice = self._prepare_tensors_for_loss(
                    mask_metric[dataset_name],
                    zeros_metric[dataset_name],
                    validation_mode=validation_mode,
                    dataset_name=dataset_name,
                    )
                    wo_bar_norm = self._compute_loss(
                    y_pred=mask_metric_full,
                    y=zeros_metric_full,
                    grid_shard_slice=grid_shard_slice,
                    dataset_name=dataset_name,
                    )
                    norm_coef = dataset_batch.shape[0] * wo_norm

            _, metrics_masked, _ = checkpoint(
                self.compute_loss_metrics,
                y_pred_metric,
                y_metric,
                step=1,
                training_mode=True,
                validation_mode=validation_mode,
                use_reentrant=False,
            )
            for key in metrics_masked.keys(): 
                if 'mse_loss' in key: 
                    metrics_masked[key] = metrics_masked[key]/norm_coef
            metrics.update(metrics_masked)

            print("WO NORM:", wo_norm)
            print("WO bar NORM:", wo_bar_norm)
            print("WO NORM + WO bar NORM =", wo_norm+wo_bar_norm)
            print("W NORMALISE = ", W_norm)
            print("METRICS RETURNED:",metrics.keys(), metrics.values())

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

    
