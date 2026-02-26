"""
利用offline RL进行训练。
"""
import sys
import torch
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
from utils.log import logger
class OfflineTrain:
    def __init__(self, dataset , dataloader , sac_model, writer , online_test ,online_test_epoch ,  batch_size, epoch , save_dir ,  config ,device='cuda'):
        self.device = device
        self.agent = sac_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.writer = writer
        self.epoch = epoch
        self.online_test = online_test
        self.online_test_epoch = online_test_epoch
        self.save_dir = save_dir
        self.config = config
        self.log_interval = int(getattr(config, "log_interval", 100))
        self.debug_lmdb_profile = bool(getattr(config, "DEBUG_LMDB_PROFILE", False))
        self.debug_lmdb_profile_every = int(getattr(config, "DEBUG_LMDB_PROFILE_EVERY", 1))
        self.num_workers = int(getattr(config, "NUM_WORKERS", 0))

    def _log_lmdb_profile(self, global_step, batch_fetch_time):
        if not self.debug_lmdb_profile:
            return
        if self.debug_lmdb_profile_every <= 0:
            return
        if global_step % self.debug_lmdb_profile_every != 0:
            return
        if self.num_workers != 0:
            tqdm.write(
                "[LMDBProfile] NUM_WORKERS>0 时 dataset 计时在子进程，主进程无法准确统计。请临时设为 0。"
            )
            return
        if not hasattr(self.dataset, "get_profile_stats"):
            return
        stats = self.dataset.get_profile_stats()
        lmdb_get_t = float(stats.get("lmdb_get_time", 0.0))
        pickle_t = float(stats.get("pickle_time", 0.0))
        getitem_t = float(stats.get("getitem_time", 0.0))
        calls = int(stats.get("getitem_calls", 0))
        if calls <= 0:
            return
        fetch_t = max(batch_fetch_time, 1e-12)
        getitem_den = max(getitem_t, 1e-12)
        tqdm.write(
            f"[LMDBProfile] step={global_step} calls={calls} "
            f"fetch={fetch_t*1000:.2f}ms get={lmdb_get_t*1000:.2f}ms "
            f"pickle={pickle_t*1000:.2f}ms "
            f"get/fetch={lmdb_get_t/fetch_t*100:.2f}% "
            f"pickle/fetch={pickle_t/fetch_t*100:.2f}% "
            f"pickle/getitem={pickle_t/getitem_den*100:.2f}%"
        )

    def train(self):
        
        global_step = 0
        for epoch in tqdm(range(1, self.epoch + 1),desc="epoch nums"):
            data_iter = iter(self.dataloader)
            for _ in range(len(self.dataloader)):
                if self.debug_lmdb_profile and hasattr(self.dataset, "reset_profile_stats"):
                    self.dataset.reset_profile_stats()
                t_fetch_start = time.perf_counter()
                batch = next(data_iter)
                batch_fetch_time = time.perf_counter() - t_fetch_start
                
                global_step += 1
                states , next_states , actions , rewards , dones = batch
                loss_dict = self.agent.update(states, actions , rewards, next_states ,  dones)
                self._log_lmdb_profile(global_step=global_step, batch_fetch_time=batch_fetch_time)

                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)
                if global_step % self.log_interval == 0:
                    tqdm.write(
                        f"step={global_step} actor_loss={loss_dict['actor_loss']:.6f} "
                        f"critic1_loss={loss_dict['critic1_loss']:.6f} critic2_loss={loss_dict['critic2_loss']:.6f}"
                    )
            if epoch % 1 == 0 :
                torch.save(self.agent.state_dict() , f'{self.save_dir}/sac_2level_model_{epoch}.pth')
            if epoch % self.online_test_epoch== 0: # 可以设置一个非常大的数进行调整曲线不进行在线测试，或者设置成使用acc进行简单的判断
                # train_acc = self.val(epoch)
                self.agent.eval()
                online_reward  , spl = self.online_test.rollout(epoch , self.agent ,logger )
                self.agent.train()
                # self.writer.add_scalar("Val/train_Accuracy", train_acc, global_step=epoch)
                self.writer.add_scalar("Val/online_reward", online_reward, global_step=epoch)
                self.writer.add_scalar("Val/spl", spl, global_step=epoch)
            if hasattr(self.dataset, "on_epoch_end"):
                self.dataset.on_epoch_end()
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=int(getattr(self.config, "NUM_WORKERS", 0)),
                    pin_memory=True,
                )
                

    def val(self, epoch):
        self.agent.eval() 

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:  
                states, next_states, actions, rewards, dones = batch
                states = states.to('cuda')
                actions = actions.to('cuda')

                actions = actions.long()  
                a_predicted_logits = self.agent.get_action(states)  

                pred_classes = torch.argmax(a_predicted_logits, dim=1)   # 预测类别
                correct = (pred_classes == actions).sum().item()         # 预测正确的数量
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.agent.train()
        return acc
    
class OfflineTrainBuffer:
    def __init__(self, dataset , dataloader , sac_model, writer , online_test ,online_test_epoch ,  batch_size, epoch , save_dir , config, device='cuda'):
        self.device = device
        self.agent = sac_model
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataloader = dataloader
        self.writer = writer
        self.epoch = epoch
        self.online_test = online_test
        self.online_test_epoch = online_test_epoch
        self.save_dir = save_dir
        self.config = config
        self.log_interval = int(getattr(config, "log_interval", 100))
        self.debug_lmdb_profile = bool(getattr(config, "DEBUG_LMDB_PROFILE", False))
        self.debug_lmdb_profile_every = int(getattr(config, "DEBUG_LMDB_PROFILE_EVERY", 1))
        self.num_workers = int(getattr(config, "NUM_WORKERS", 0))

    def _log_lmdb_profile(self, global_step, batch_fetch_time):
        if not self.debug_lmdb_profile:
            return
        if self.debug_lmdb_profile_every <= 0:
            return
        if global_step % self.debug_lmdb_profile_every != 0:
            return
        if self.num_workers != 0:
            tqdm.write(
                "[LMDBProfile] NUM_WORKERS>0 时 dataset 计时在子进程，主进程无法准确统计。请临时设为 0。"
            )
            return
        if not hasattr(self.dataset, "get_profile_stats"):
            return
        stats = self.dataset.get_profile_stats()
        lmdb_get_t = float(stats.get("lmdb_get_time", 0.0))
        pickle_t = float(stats.get("pickle_time", 0.0))
        getitem_t = float(stats.get("getitem_time", 0.0))
        calls = int(stats.get("getitem_calls", 0))
        if calls <= 0:
            return
        fetch_t = max(batch_fetch_time, 1e-12)
        getitem_den = max(getitem_t, 1e-12)
        tqdm.write(
            f"[LMDBProfile] step={global_step} calls={calls} "
            f"fetch={fetch_t*1000:.2f}ms get={lmdb_get_t*1000:.2f}ms "
            f"pickle={pickle_t*1000:.2f}ms "
            f"get/fetch={lmdb_get_t/fetch_t*100:.2f}% "
            f"pickle/fetch={pickle_t/fetch_t*100:.2f}% "
            f"pickle/getitem={pickle_t/getitem_den*100:.2f}%"
        )
    def train(self):
        
        global_step = 0
        for epoch in tqdm(range(1, self.epoch + 1),desc="epoch nums"):
            if epoch % self.config.replay_epoch == 0 and epoch != 0 and epoch < 10:
                self.dataset.replay(logger)
                self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True ,  pin_memory=True)
            data_iter = iter(self.dataloader)
            for _ in range(len(self.dataloader)):
                if self.debug_lmdb_profile and hasattr(self.dataset, "reset_profile_stats"):
                    self.dataset.reset_profile_stats()
                t_fetch_start = time.perf_counter()
                batch = next(data_iter)
                batch_fetch_time = time.perf_counter() - t_fetch_start
                
                global_step += 1
                states , next_states , actions , rewards , dones = batch

                loss_dict = self.agent.update(states, actions , rewards, next_states ,  dones)
                self._log_lmdb_profile(global_step=global_step, batch_fetch_time=batch_fetch_time)

                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)
                if global_step % self.log_interval == 0:
                    tqdm.write(
                        f"step={global_step} actor_loss={loss_dict['actor_loss']:.6f} "
                        f"critic1_loss={loss_dict['critic1_loss']:.6f} critic2_loss={loss_dict['critic2_loss']:.6f}"
                    )
            if epoch % 1 == 0 :
                torch.save(self.agent.state_dict() , f'{self.save_dir}/sac_2level_model_{epoch}.pth')
            if epoch % self.online_test_epoch== 0: # 可以设置一个非常大的数进行调整曲线不进行在线测试，或者设置成使用acc进行简单的判断
                # train_acc = self.val(epoch)
                self.agent.eval()
                online_reward  , spl = self.online_test.rollout(epoch , self.agent ,logger )
                self.agent.train()
                # self.writer.add_scalar("Val/train_Accuracy", train_acc, global_step=epoch)
                self.writer.add_scalar("Val/online_reward", online_reward, global_step=epoch)
                self.writer.add_scalar("Val/spl", spl, global_step=epoch)
            if hasattr(self.dataset, "on_epoch_end"):
                self.dataset.on_epoch_end()
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=int(getattr(self.config, "NUM_WORKERS", 0)),
                    pin_memory=True,
                )
                

    def val(self, epoch):
        self.agent.eval() 

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:  
                states, next_states, actions, rewards, dones = batch
                states = states.to('cuda')
                actions = actions.to('cuda')

                actions = actions.long()  
                a_predicted_logits = self.agent.get_action(states)  

                pred_classes = torch.argmax(a_predicted_logits, dim=1)   # 预测类别
                correct = (pred_classes == actions).sum().item()         # 预测正确的数量
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.agent.train()
        return acc
    
class OfflineAndHybird:
    def __init__(self, dataset , dataloader , sac_model, writer , online_test ,online_test_epoch ,  batch_size, epoch , save_dir , config, device='cuda'):
        self.device = device
        self.agent = sac_model
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataloader = dataloader
        self.writer = writer
        self.epoch = epoch
        self.online_test = online_test
        self.online_test_epoch = online_test_epoch
        self.save_dir = save_dir
        self.config = config
        self.log_interval = int(getattr(config, "log_interval", 100))
        self.debug_lmdb_profile = bool(getattr(config, "DEBUG_LMDB_PROFILE", False))
        self.debug_lmdb_profile_every = int(getattr(config, "DEBUG_LMDB_PROFILE_EVERY", 1))
        self.num_workers = int(getattr(config, "NUM_WORKERS", 0))

    def _log_lmdb_profile(self, global_step, batch_fetch_time):
        # if not self.debug_lmdb_profile:
        #     return
        # if self.debug_lmdb_profile_every <= 0:
        #     return
        # if global_step % self.debug_lmdb_profile_every != 0:
        #     return
        # if self.num_workers != 0:
        #     tqdm.write(
        #         "[LMDBProfile] NUM_WORKERS>0 时 dataset 计时在子进程，主进程无法准确统计。请临时设为 0。"
        #     )
        #     return
        # if not hasattr(self.dataset, "get_profile_stats"):
        #     return
        stats = self.dataset.get_profile_stats()
        lmdb_get_t = float(stats.get("lmdb_get_time", 0.0))
        pickle_t = float(stats.get("pickle_time", 0.0))
        getitem_t = float(stats.get("getitem_time", 0.0))
        calls = int(stats.get("getitem_calls", 0))
        if calls <= 0:
            return
        fetch_t = max(batch_fetch_time, 1e-12)
        getitem_den = max(getitem_t, 1e-12)
        tqdm.write(
            f"[LMDBProfile] step={global_step} calls={calls} "
            f"fetch={fetch_t*1000:.2f}ms get={lmdb_get_t*1000:.2f}ms "
            f"pickle={pickle_t*1000:.2f}ms "
            f"get/fetch={lmdb_get_t/fetch_t*100:.2f}% "
            f"pickle/fetch={pickle_t/fetch_t*100:.2f}% "
            f"pickle/getitem={pickle_t/getitem_den*100:.2f}%"
        )
    def train(self):
        
        global_step = 0
        for epoch in tqdm(range(1, self.epoch + 1),desc="epoch nums"):
            if epoch % self.config.replay_epoch == 0 and epoch != 0 and epoch < 50:
                self.dataset.replay(logger)
                self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True ,  pin_memory=True)
            data_iter = iter(self.dataloader)
            for _ in range(len(self.dataloader)):
                if self.debug_lmdb_profile and hasattr(self.dataset, "reset_profile_stats"):
                    self.dataset.reset_profile_stats()
                t_fetch_start = time.perf_counter()
                batch = next(data_iter)
                batch_fetch_time = time.perf_counter() - t_fetch_start
                
                global_step += 1

                # batch_audio , batch_rgb ,batch_depth, batch_angle , batch_action = batch
                states , next_states , actions , rewards , dones = batch

                loss_dict = self.agent.update(states, actions , rewards, next_states ,  dones)
                self._log_lmdb_profile(global_step=global_step, batch_fetch_time=batch_fetch_time)

                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"scalar/{key}", value, global_step=global_step)
                # if global_step % self.log_interval == 0:
                tqdm.write(
                        f"step={global_step} actor_loss={loss_dict['actor_loss']:.6f} "
                        f"critic1_loss={loss_dict['critic1_loss']:.6f} critic2_loss={loss_dict['critic2_loss']:.6f}"
                    )
            # if epoch % 10 == 0 :
            #     torch.save(self.agent.state_dict() , f'{self.save_dir}/sac_2level_model_{epoch}.pth')
            # if epoch % self.online_test_epoch== 0: # 可以设置一个非常大的数进行调整曲线不进行在线测试，或者设置成使用acc进行简单的判断
            #     # train_acc = self.val(epoch)
            #     self.agent.eval()
            #     online_reward  , spl = self.online_test.rollout(epoch , self.agent ,logger )
            #     self.agent.train()
            #     # self.writer.add_scalar("Val/train_Accuracy", train_acc, global_step=epoch)
            #     self.writer.add_scalar("Val/online_reward", online_reward, global_step=epoch)
            #     self.writer.add_scalar("Val/spl", spl, global_step=epoch)
            if hasattr(self.dataset, "on_epoch_end"):
                self.dataset.on_epoch_end()
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=int(getattr(self.config, "NUM_WORKERS", 0)),
                    pin_memory=True,
                )
                

    def val(self, epoch):
        self.agent.eval() 

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:  
                states, next_states, actions, rewards, dones = batch
                states = states.to('cuda')
                actions = actions.to('cuda')

                actions = actions.long()  
                a_predicted_logits = self.agent.get_action(states)  

                pred_classes = torch.argmax(a_predicted_logits, dim=1)   # 预测类别
                correct = (pred_classes == actions).sum().item()         # 预测正确的数量
                total_correct += correct
                total_samples += actions.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0.0
        self.agent.train()
        return acc
