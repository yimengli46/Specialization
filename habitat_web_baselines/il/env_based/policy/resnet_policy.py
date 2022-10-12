from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from gym.spaces import Dict, Box
from habitat import Config, logger
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoalSensor,
    #task_cat2mpcat40,
    #mapping_mpcat40_to_goal21
)
from habitat_web_baselines.il.common.encoders.resnet_encoders import (
    VlnResnetDepthEncoder,
    ResnetRGBEncoder,
    ResnetSemSeqEncoder,
)
from habitat_web_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_web_baselines.common.baseline_registry import baseline_registry
from habitat_web_baselines.rl.ppo import Net, Policy

task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    5,  # ('table', 4, 1)
    6,  # ('picture', 5, 2)
    7,  # ('cabinet', 6, 3)
    8,  # ('cushion', 7, 4)
    10,  # ('sofa', 9, 5),
    11,  # ('bed', 10, 6)
    13,  # ('chest_of_drawers', 12, 7),
    14,  # ('plant', 13, 8)
    15,  # ('sink', 14, 9)
    18,  # ('toilet', 17, 10),
    19,  # ('stool', 18, 11),
    20,  # ('towel', 19, 12)
    22,  # ('tv_monitor', 21, 13)
    23,  # ('shower', 22, 14)
    25,  # ('bathtub', 24, 15)
    26,  # ('counter', 25, 16),
    27,  # ('fireplace', 26, 17),
    33,  # ('gym_equipment', 32, 18),
    34,  # ('seating', 33, 19),
    38,  # ('clothes', 37, 20),
    43,  # ('foodstuff', 42, 21),
    44,  # ('stationery', 43, 22),
    45,  # ('fruit', 44, 23),
    46,  # ('plaything', 45, 24),
    47,  # ('hand_tool', 46, 25),
    48,  # ('game_equipment', 47, 26),
    49,  # ('kitchenware', 48, 27)
]

mapping_mpcat40_to_goal21 = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}



class ObjectNavILNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions, device=None):
        super().__init__()
        self.model_config = model_config
        rnn_input_size = 0

        # Init the depth encoder
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=128,
            checkpoint="model_weights/gibson-2plus-resnet50.pth",
            backbone="resnet50",
            trainable=False,
        )
        rnn_input_size += 128


        # Init the RGB visual encoder
        self.rgb_encoder = ResnetRGBEncoder(
            observation_space,
            output_size=256,
            backbone="resnet18",
            trainable=True,
            normalize_visual_inputs=False,
        )
        rnn_input_size += 256

        sem_seg_output_size = 0
        self.semantic_predictor = None
        self.is_thda = False
        if True:
            sem_embedding_size = 4
            self.is_thda = True
            rgb_shape = observation_space.spaces["rgb"].shape
            spaces = {
                "semantic": Box(
                    low=0,
                    high=255,
                    shape=(rgb_shape[0], rgb_shape[1], sem_embedding_size),
                    dtype=np.uint8,
                ),
            }
            sem_obs_space = Dict(spaces)
            self.sem_seg_encoder = ResnetSemSeqEncoder(
                sem_obs_space,
                output_size=256,
                backbone="resnet18",
                trainable=True,
                semantic_embedding_size=sem_embedding_size,
                is_thda=self.is_thda
            )
            sem_seg_output_size = 256
            logger.info("Setting up Sem Seg model")
            rnn_input_size += sem_seg_output_size

            self.embed_sge = True
            if self.embed_sge:
                self.task_cat2mpcat40 = torch.tensor(task_cat2mpcat40, device=device)
                self.mapping_mpcat40_to_goal = np.zeros(
                    max(
                        max(mapping_mpcat40_to_goal21.keys()) + 1,
                        50,
                    ),
                    dtype=np.int8,
                )

                for key, value in mapping_mpcat40_to_goal21.items():
                    self.mapping_mpcat40_to_goal[key] = value
                self.mapping_mpcat40_to_goal = torch.tensor(self.mapping_mpcat40_to_goal, device=device)
                rnn_input_size += 1

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")
        
        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(input_compass_dim, self.compass_embedding_dim)
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            if self.is_thda:
                self._n_object_categories = 28
            logger.info("Object categories: {}".format(self._n_object_categories))
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if True:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.rnn_input_size = rnn_input_size

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=2048,
            num_layers=2,
            rnn_type="GRU",
        )

        self.train()

    @property
    def output_size(self):
        return 2048

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _extract_sge(self, observations):
        # recalculating to keep this self-contained instead of depending on training infra
        if "semantic" in observations and "objectgoal" in observations:
            obj_semantic = observations["semantic"].contiguous().flatten(start_dim=1)
            
            if len(observations["objectgoal"].size()) == 3:
                observations["objectgoal"] = observations["objectgoal"].contiguous().view(
                    -1, observations["objectgoal"].size(2)
                )

            idx = self.task_cat2mpcat40[
                observations["objectgoal"].long()
            ]
            if self.is_thda:
                idx = self.mapping_mpcat40_to_goal[idx].long()
            idx = idx.to(obj_semantic.device)

            if len(idx.size()) == 3:
                idx = idx.squeeze(1)

            goal_visible_pixels = (obj_semantic == idx).sum(dim=1)
            goal_visible_area = torch.true_divide(goal_visible_pixels, obj_semantic.size(-1)).float()
            return goal_visible_area.unsqueeze(-1)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]
        depth_obs = observations["depth"]

        x = []

        if self.depth_encoder is not None:
            if len(depth_obs.size()) == 5:
                observations["depth"] = depth_obs.contiguous().view(
                    -1, depth_obs.size(2), depth_obs.size(3), depth_obs.size(4)
                )

            depth_embedding = self.depth_encoder(observations)
            x.append(depth_embedding)

        if self.rgb_encoder is not None:
            if len(rgb_obs.size()) == 5:
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )

            rgb_embedding = self.rgb_encoder(observations)
            x.append(rgb_embedding)

        if True:
            semantic_obs = observations["semantic"]
            if len(semantic_obs.size()) == 4:
                observations["semantic"] = semantic_obs.contiguous().view(
                    -1, semantic_obs.size(2), semantic_obs.size(3)
                )
            if self.embed_sge:
                sge_embedding = self._extract_sge(observations)
                x.append(sge_embedding)

            sem_seg_embedding = self.sem_seg_encoder(observations)
            x.append(sem_seg_embedding)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))
        
        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(-1, obs_compass.size(2))
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations.squeeze(dim=1))
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(-1, object_goal.size(2))
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if True:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x.append(prev_actions_embedding)
        
        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


@baseline_registry.register_policy
class ObjectNavILPolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            ObjectNavILNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
            no_critic=True
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=None,            
        )
