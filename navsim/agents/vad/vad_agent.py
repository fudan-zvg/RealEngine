from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent, Trajectory
from navsim.agents.vad.vad_config import VADConfig
from navsim.agents.vad.vad_model import VADModel
from navsim.agents.vad.vad_loss import vad_loss
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.vad.vad_features import VADFeatureBuilder, VADTargetBuilder, VADTransfuserTargetBuilder, vad_collate
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class VADAgent(AbstractAgent):
    """Agent interface for VAD baseline."""

    def __init__(
        self,
        config: VADConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes VAD agent.
        :param config: global config of VAD agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._vad_model = VADModel(config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [VADTransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        # return [VADFeatureBuilder(config=self._config)]
        return [VADFeatureBuilder(config=self._config)]

    def get_collate_function(self):
        return vad_collate

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._vad_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(features, targets, predictions, self._config)
    
    def compute_trajectory(self, agent_input):
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        new_features = {}
        for k, v in features.items():
            if k == 'metas':
                new_features[k] = [v]
            else:
                new_features[k] = v.unsqueeze(0)
        features = new_features

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._vad_model.parameters(), lr=self._lr)
