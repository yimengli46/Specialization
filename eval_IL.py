import numpy as np
import torch
from core.config import get_config
import habitat
from habitat_web_baselines.il.env_based.policy.resnet_policy import ObjectNavILNet

from habitat_web_baselines.il.env_based.algos.agent import ILAgent

from habitat_web_baselines.il.env_based.policy.rednet import load_rednet
from habitat_web_baselines.common.baseline_registry import baseline_registry
from habitat_web_baselines.utils.common import batch_obs

from typing import Any, DefaultDict, Dict, List, Optional
import tqdm

from habitat_web_baselines.common.environments import get_env_class, NavRLEnv

METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "room_visitation_map", "exploration_metrics"}

def extract_scalars_from_info(info):
        result = {}
        for k, v in info.items():
            if k in METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in extract_scalars_from_info(v).items()
                        if (k + "." + subk) not in METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

device = torch.device('cuda:0')

cfg = get_config('configs/exp_specialization.yaml', None)


#======================== initialize habitat environment ========================
#config = habitat.get_config(config_paths=cfg.GENERAL.OBJECTNAV_HABITAT_CONFIG_PATH)
cfg.defrost()
cfg.TASK_CONFIG.DATASET.SPLIT = 'val'
cfg.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
cfg.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
cfg.freeze()

#env = habitat.Env(config=config)
#env = get_env_class('NavRLEnv')
env = NavRLEnv(config=cfg)

#=============================== set up actor critic agent =================================
observation_space = env.observation_space
#policy = ObjectNavILNet(observation_space, None, env.action_space.n)
policy = baseline_registry.get_policy("ObjectNavILPolicy")
policy = policy.from_config(None, observation_space, env.action_space)
policy.to(device)

semantic_predictor = None
semantic_predictor = load_rednet(device, ckpt="model_weights/rednet_semmap_mp3d_tuned.pth", resize=True, num_classes=29)
semantic_predictor.eval()

agent = ILAgent(model=policy, num_envs=1, num_mini_batch=1, lr=0.001, eps=1.0e-5, max_grad_norm=0.2)

#=============================== import torch model ===========================
ckpt_path = 'model_weights/objectnav_semseg.ckpt'
agent.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'], strict=True)
 
policy = agent.model
policy.eval()

stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode

#==================================== run the model ==============================
number_of_eval_episodes = 5
if number_of_eval_episodes == -1:
	number_of_eval_episodes = sum(env.number_of_episodes)
else:
	total_num_eps = env.number_of_episodes
	if total_num_eps < number_of_eval_episodes:
		print(f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {total_num_eps}.")
		print(f"Evaluating with {total_num_eps} instead.")
		number_of_eval_episodes = total_num_eps

pbar = tqdm.tqdm(total=number_of_eval_episodes)
while len(stats_episodes) < number_of_eval_episodes:
	observation = env.reset()

	batch = batch_obs([observation], device=device)
	#batch = apply_obs_transforms_batch(batch, self.obs_transforms)

	current_episode_reward = torch.zeros(1, 1, device=device)

	test_recurrent_hidden_states = torch.zeros(2, 1, 2048, device=device)
	prev_actions = torch.zeros(1, 1, device=device, dtype=torch.long)
	not_done_masks = torch.zeros(1, 1, device=device)

	current_episode_steps = torch.zeros(1, 1, device=device)

	current_episode = env.current_episode

	while  True:
		
		with torch.no_grad():
			if semantic_predictor is not None:
				batch["semantic"] = semantic_predictor(batch["rgb"], batch["depth"])
				batch["semantic"] = batch["semantic"] - 1
			
			logits, test_recurrent_hidden_states = policy(batch, test_recurrent_hidden_states, prev_actions,
				not_done_masks)
			current_episode_steps += 1

			actions = torch.argmax(logits, dim=1)
			prev_actions.copy_(actions.unsqueeze(1))  # type: ignore

		# NB: Move actions to CPU.  If CUDA tensors are
		# sent in to env.step(), that will create CUDA contexts
		# in the subprocesses.
		# For backwards compatibility, we also call .item() to convert to
		# an int
		step_data = [a.item() for a in actions.to(device="cpu")]
		# not sure if this prev_act_dict is gonna affect the output
		prev_act_dict = {'action': 2}
		output = [env.step(*step_data, **prev_act_dict)]
		#print(f'output = {output}')


		observation, reward_l, done, info = [list(x) for x in zip(*output)]
		#print(f'rewards_l = {reward_l}')
		#print(f'dones = {done}')
		#print(f'infos = {info}')
		#assert 1==2
		batch = batch_obs(observation, device=device)
		#batch = apply_obs_transforms_batch(batch, self.obs_transforms)

		not_done_masks = torch.tensor([[0.0] if done[0] else [1.0]], dtype=torch.float, device=device)

		reward = torch.tensor(reward_l, dtype=torch.float, device=device).unsqueeze(1)
		print(f'*********rewards = {reward}')
		current_episode_reward += reward
		
		# episode ended
		if int(not_done_masks[0].item()) == 0:
			pbar.update()
			episode_stats = {}
			episode_stats["reward"] = current_episode_reward[0].item()
			episode_stats.update(
				extract_scalars_from_info(info[0])
			)
			current_episode_reward[0] = 0
			current_episode_steps[0] = 0

			# use scene_id + episode_id as unique id for storing stats
			stats_episodes[(current_episode.scene_id, current_episode.episode_id)] = episode_stats

			break


num_episodes = len(stats_episodes)
aggregated_stats = {}
for stat_key in next(iter(stats_episodes.values())).keys():
	aggregated_stats[stat_key] = (
		sum(v[stat_key] for v in stats_episodes.values())
		/ num_episodes
	)

for k, v in aggregated_stats.items():
	print(f"Average episode {k}: {v:.4f}")

'''
step_id = checkpoint_index
if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
	step_id = ckpt_dict["extra_state"]["step"]

writer.add_scalars(
	"eval_reward",
	{"average reward": aggregated_stats["reward"]},
	step_id,
)

metrics = {k: v for k, v in aggregated_stats.items() if k not in ["reward", "pred_reward"]}
if len(metrics) > 0:
	writer.add_scalars("eval_metrics", metrics, step_id)
'''

env.close()
