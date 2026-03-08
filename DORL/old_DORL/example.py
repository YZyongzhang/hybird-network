from DORL.Dataset_env import build_env_from_dorl_pt
from DORL.train_rl import DORLSACTrainer, DORLTrainConfig

# 1) 从 Generate_DORL_PT 生成的 shard 目录加载 episode env
env = build_env_from_dorl_pt(
    root="media/pt/offline_muti_embedding_DORL",
    random_episode=True,
)
obs = env.reset()
next_obs, reward, done, info = env.step(action=1)
print("one step:", reward, done, info)

# 2) 训练一个离散 SAC（DORL）
cfg = DORLTrainConfig(
    action_dim=4,
    train_epochs=30,
    episodes_per_epoch=200,
    max_steps=200,
    batch_size=256,
    warmup_steps=2000,
    ckpt_dir="media/DORL/ckpt",
)
trainer = DORLSACTrainer(env=env, config=cfg)
history = trainer.train()
print("last epoch:", history[-1])
