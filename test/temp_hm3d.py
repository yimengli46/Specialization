import habitat_sim

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "/home/yimeng/work/topo_map_specialization/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
backend_cfg.scene_dataset_config_file = "/home/yimeng/work/topo_map_specialization/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sem_cfg]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)