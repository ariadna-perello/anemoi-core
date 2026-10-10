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
from anemoi.training.losses.scalers.node_attributes import ObsMaskGraphNodeAttributeScaler
from anemoi.training.losses.scalers.base_scaler import AvailableCallbacks
 
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
        self.obs_mask = None

    '''

    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], _: int) -> dict[str, torch.Tensor]:
        """Override to compute observation mask before scaler updates."""
        
        for dataset_name, dataset_batch in batch.items():
            if dataset_name == "innovations": 
                innovations = self.model.post_processors[dataset_name](dataset_batch[:,:,:self.n_step_input],in_place=False)
                self.obs_mask = innovations == 0
                print("POINTS D'OBS TRAIN:", torch.count_nonzero(~self.obs_mask))

            if dataset_name == "observations": 
                obs = self.model.post_processors[dataset_name](dataset_batch[:,:,:self.n_step_input],in_place=False)
                self.obs_mask = torch.isnan(obs)
                print("POINTS D'OBS TRAIN:", torch.count_nonzero(~self.obs_mask))
        
        return super().on_after_batch_transfer(batch, _)
    '''

    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], _: int) -> dict[str, torch.Tensor]:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.

        Also normalize the batch in-place if needed.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch to transfer

        Returns
        -------
        dict[str, torch.Tensor]
            Batch after transfer
        """
        assert isinstance(batch, dict), "batch must be a dict keyed by dataset name"
        # Gathering/sharding of batch
        batch = self._setup_batch_sharding(batch)

        # Batch normalization
        batch = self._normalize_batch(batch)

        for dataset_name, dataset_batch in batch.items():
            if dataset_name == "innovations": 
                innovations = self.model.post_processors[dataset_name](dataset_batch[:,:,:self.n_step_input],in_place=False)
                self.obs_mask = innovations == 0
                print("POINTS D'OBS TRAIN:", torch.count_nonzero(~self.obs_mask))

            if dataset_name == "observations": 
                obs = self.model.post_processors[dataset_name](dataset_batch[:,:,:self.n_step_input],in_place=False)
                self.obs_mask = torch.isnan(obs)
                print("POINTS D'OBS TRAIN:", torch.count_nonzero(~self.obs_mask))

        # Prepare scalers, e.g. init delayed scalers and update scalers
        self._prepare_loss_scalers()

        return super().on_after_batch_transfer(batch, _)

    def update_scalers(self, callback: AvailableCallbacks) -> None:
        """Update scalers, calling the defined function on them, updating if not None."""
        print("UPDATE SCALERS OBS MASK:", self.obs_mask)
        for dataset_name, dataset_scalers in self.updating_scalars.items():
            for name, scaler_builder in dataset_scalers.items():
                if isinstance(scaler_builder, ObsMaskGraphNodeAttributeScaler):
                    if self.obs_mask is not None: 
                        scaler_builder.set_obs_mask(self.obs_mask)
                self._update_scaler_for_dataset(
                    name,
                    scaler_builder,
                    callback,
                    self.loss[dataset_name],
                    self.metrics[dataset_name],
                    dataset_name=dataset_name,
                )



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

            if validation_mode: 
                print(dataset_name, "X, number of nans:", torch.count_nonzero(torch.isnan(x[dataset_name])))
            
            #if dataset_name == "innovations": 
            #    innovations = self.model.post_processors[dataset_name](x[dataset_name],in_place=False)
            #    self.obs_mask = innovations == 0
            #    print("POINTS D'OBS TRAIN:", torch.count_nonzero(~self.obs_mask))

            #if dataset_name == "observations": 
            #    obs = self.model.post_processors[dataset_name](x[dataset_name],in_place=False)
            #    self.obs_mask = torch.isnan(obs)
            #    print("POINTS D'OBS TRAIN:", torch.count_nonzero(~self.obs_mask))
                
            
        y_pred = self(x)
        #print("SHAPE DE Y_PRED:",y_pred[dataset_name].shape)


        #print("X shape: ", x["increments_input"].shape)
        #print("Y shape: ", y["increments_target"].shape)

        #mse = ((x["increments_input"]-y["increments_target"])**2).mean()
        #mse = torch.nn.MSELoss()(x["increments_input"],y["increments_target"])
        #print("MSE ENTRE X ET Y:",mse)
        #print("X ET Y ÉGAUX?", torch.equal(x["increments_input"],y["increments_target"]))

        # y includes the auxiliary variables, so we must leave those out when computing the loss


        y = {}
        y_metric = {}
        y_pred_metric = {}

        for dataset_name, dataset_batch in batch.items():
            print("Y dataset name:", dataset_name)
            y[dataset_name] = dataset_batch[:,-1][...,self.data_indices[dataset_name].data.output.full]
            LOGGER.info(f"Shape output : {y[dataset_name].shape}  for {dataset_name}")
            if len(y[dataset_name].shape)<5: 
                LOGGER.info ("unsqueeze data output")
                y[dataset_name] = y[dataset_name].unsqueeze(2)

            y_metric[dataset_name] = y[dataset_name].clone().detach()
            y_pred_metric[dataset_name] = y_pred[dataset_name].clone().detach()
            if validation_mode: 
                print(dataset_name, "Y, number of nans:", torch.count_nonzero(torch.isnan(y[dataset_name])))
                print(dataset_name, "Y_pred, number of nans:", torch.count_nonzero(torch.isnan(y_pred[dataset_name])))


        loss, metrics, y_pred = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            step=0,
            training_mode=True,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        if validation_mode and self.obs_mask is not None: 
            for dataset_name, dataset_batch in batch.items():
                print("Y dataset name:", dataset_name)
                if len(self.data_indices[dataset_name].data.output.full) != 0
                    y_metric[dataset_name][self.obs_mask] = 0
                    y_pred_metric[dataset_name][self.obs_mask] = 0
                    
            
            _, metrics_masked, _ = checkpoint(
                self.compute_loss_metrics,
                y_pred_metric,
                y_metric,
                step=1,
                training_mode=True,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            metrics.update(metrics_masked)

        # All tasks return (loss, metrics, list of per-step dicts) for consistent plot callback contract.
        if validation_mode: 
            print("VALIDATION MODE")
            print("METRICS RETURNED:",metrics.keys(), metrics.values())
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

    
