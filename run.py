import torch
import os

from configs.default import get_config
from env import Env
from utils.log import append_experiment_journal, logger, setup_run_logger


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_hybrid_model(ckpt_path):
    from network import HybirdNetwork

    device = _get_device()
    logger.info("loading hybrid model ckpt: %s", ckpt_path)
    model = HybirdNetwork().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    logger.info("hybrid model loaded on device=%s", device)
    return model


def _run_collect(config, collect_config):
    from col import COLLECTER
    from run import Collect

    logger.info("task=COLLECT type=%s", collect_config.TYPE)
    logger.info("collect config: %s", collect_config)
    env = Env(config=config)
    model = _load_hybrid_model(collect_config.COLLECT_CKPT)
    collecter = COLLECTER(config, env, model=model)
    logger.info("collect begin")
    Collect(collecter=collecter)
    logger.info("collect finished")


def _run_pt(pt_config):
    from train.VADE import LoadLmdb

    logger.info("task=PT type=%s", pt_config.TYPE)
    logger.info("pt config: %s", pt_config)
    if pt_config.TYPE == "HybirdNetwork":
        LoadLmdb.load_pt(pt_config.RAW_DATA_PATH, pt_config)
        return
    if pt_config.TYPE == "OfflineWithHybridPT":
        LoadLmdb.load_offline_with_hybrid(pt_config.RAW_DATA_PATH, config=pt_config)
        return
    if pt_config.TYPE == "HybirdNetworkTwoFrame":
        LoadLmdb.load_two_frame_pt(pt_config.RAW_DATA_PATH, pt_config)
        return

    # Types below require foundation ckpt.
    ckpt_required_types = {
        "OfflineTwoFrameWithHybridPT": "load_offline_two_frame",
        "offline": "load_offline",
        "offlinetwoframe": "load_offline_two_frame",
        "offlinelstm": "load_offline_lstm",
        "offlinelstm_v15": "load_offline_lstm_v15",
        "offlinelstm_by_level": "load_offline_lstm_level",
        "offlinelstm_by_level_audio_visual": "load_offline_lstm_level_audio_visual",
    }
    if pt_config.TYPE not in ckpt_required_types:
        raise ValueError(f"Unsupported PT.TYPE: {pt_config.TYPE}")

    model = _load_hybrid_model(pt_config.CKPT)
    method_name = ckpt_required_types[pt_config.TYPE]
    method = getattr(LoadLmdb, method_name)
    logger.info("pt conversion method=%s raw_data=%s to_path=%s", method_name, pt_config.RAW_DATA_PATH, pt_config.TO_PATH)
    method(pt_config.RAW_DATA_PATH, model=model, config=pt_config)
    logger.info("pt conversion finished")


def _run_eval(config, eval_config):
    logger.info("task=EVAL type=%s", eval_config.TYPE)
    logger.info("eval config: %s", eval_config)
    if eval_config.TYPE == "OfflineRL_v1_3":
        from tools.eval_hybrid_sac import run_v1_3_eval

        run_v1_3_eval(
            config=config,
            hybrid_ckpt=eval_config.HYBRID_CKPT,
            sac_ckpt=eval_config.SAC_CKPT,
            episodes=eval_config.EPISODES,
            max_steps=eval_config.MAX_STEPS,
            seed=eval_config.SEED,
            stochastic_sac=eval_config.STOCHASTIC_SAC,
        )
        logger.info("eval finished")
        return
    if eval_config.TYPE == "OfflineRL_v1_5":
        from tools.eval_hybrid_sac import run_v1_5_eval

        run_v1_5_eval(
            config=config,
            hybrid_ckpt=eval_config.HYBRID_CKPT,
            sac_ckpt=eval_config.SAC_CKPT,
            episodes=eval_config.EPISODES,
            max_steps=eval_config.MAX_STEPS,
            seed=eval_config.SEED,
            stochastic_sac=eval_config.STOCHASTIC_SAC,
            temporal_ckpt=getattr(eval_config, "TEMPORAL_CKPT", ""),
            actor_ckpt=getattr(eval_config, "ACTOR_CKPT", ""),
            critic1_ckpt=getattr(eval_config, "CRITIC1_CKPT", ""),
            critic2_ckpt=getattr(eval_config, "CRITIC2_CKPT", ""),
        )
        logger.info("eval finished")
        return
    raise ValueError(f"Unsupported EVAL.TYPE: {eval_config.TYPE}")


def _build_offline_agent(config, offline_config):
    from train import OnlineTest

    env = Env(config)
    hybrid_model = _load_hybrid_model(offline_config.ONLINE_CKPT)
    online_test = OnlineTest(env=env, hybirdmodel=hybrid_model, config=offline_config)

    state_dim = offline_config.state_dim
    action_dim = offline_config.action_dim
    hidden_dim = offline_config.hidden_dim
    lr = offline_config.lr
    tau = offline_config.tau
    gamma = offline_config.gamma
    beta = offline_config.beta
    target_entropy = offline_config.target_entropy
    device = _get_device()

    if offline_config.model == "v1":
        from network import SAC_model
        from train import OfflineTrain, OfflineTrainBuffer

        sac_model = SAC_model(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=lr,
            critic_lr=lr,
            alpha_lr=lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
        )
        trainer = OfflineTrainBuffer if offline_config.buffer else OfflineTrain
    elif offline_config.model == "v1_3":
        from network import SAC_Hybird_model
        from train import OfflineAndHybird

        sac_model = SAC_Hybird_model(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=lr,
            critic_lr=lr,
            alpha_lr=lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            hybird_ckpt_path=offline_config.ONLINE_CKPT,
        )
        trainer = OfflineAndHybird
    elif offline_config.model == "v1_4":
        from network import SAC_Hybird_LSTM_CQL_model
        from train import OfflineAndHybird

        lstm_hidden_dim = int(getattr(offline_config, "lstm_hidden_dim", state_dim))
        lstm_num_layers = int(getattr(offline_config, "lstm_num_layers", 1))
        sac_model = SAC_Hybird_LSTM_CQL_model(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=lr,
            critic_lr=lr,
            alpha_lr=lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            hybird_ckpt_path=offline_config.ONLINE_CKPT,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        trainer = OfflineAndHybird
    elif offline_config.model == "v1_5":
        from network import SAC_LSTM_CQL_v1_5
        from train import OfflineTrain

        lstm_hidden_dim = int(getattr(offline_config, "lstm_hidden_dim", state_dim))
        lstm_num_layers = int(getattr(offline_config, "lstm_num_layers", 1))
        sac_model = SAC_LSTM_CQL_v1_5(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=lr,
            critic_lr=lr,
            alpha_lr=lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            beta=beta,
            device=device,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
        )
        trainer = OfflineTrain
    elif offline_config.model == "v2":
        from network import CQLSAC
        from train import OfflineTrain, OfflineTrainBuffer

        sac_model = CQLSAC(
            state_size=state_dim,
            action_size=action_dim,
            hidden_size=hidden_dim,
            beta=beta,
            device=device,
        )
        trainer = OfflineTrainBuffer if offline_config.buffer else OfflineTrain
    elif offline_config.model == "v4":
        from network import cql_lstm
        from train import OfflineTrain

        sac_model = cql_lstm(
            state_size=state_dim,
            action_size=action_dim,
            tau=tau,
            hidden_size=hidden_dim,
            learning_rate=lr,
            with_lagrange=False,
            target_action_gap=0,
            device=device,
            lstm_seq_len=5,
            lstm_layer=1,
            lstm_out=128,
        )
        trainer = OfflineTrain
    elif offline_config.model == "v5":
        from network import cql_lstm_attention
        from train import OfflineTrain

        sac_model = cql_lstm_attention(
            state_size=state_dim,
            action_size=action_dim,
            tau=tau,
            hidden_size=hidden_dim,
            learning_rate=lr,
            with_lagrange=False,
            target_action_gap=0,
            device=device,
            lstm_seq_len=5,
            lstm_layer=1,
            lstm_out=128,
        )
        trainer = OfflineTrain
    else:
        raise ValueError(f"Unsupported OFFLINE.model: {offline_config.model}")

    if offline_config.LOAD_PATH:
        logger.info("loading offline agent ckpt: %s", offline_config.MODEL_PATH)
        sac_model.load_state_dict(torch.load(offline_config.MODEL_PATH, map_location=device))
        sac_model.train()

    logger.info("offline agent built: model=%s trainer=%s", offline_config.model, trainer.__name__)
    return sac_model, trainer, online_test


def _run_train(config, train_config):
    from run import Train

    logger.info("task=TRAIN type=%s", train_config.TYPE)
    logger.info("train config: %s", train_config)
    if train_config.TYPE == "HybirdNetworkAudio":
        from network import AudioCRNN
        from train import HybirdNetworkAudioTrain

        model = AudioCRNN()
        Train(model=model, trainer=HybirdNetworkAudioTrain, config=train_config)
        logger.info("train finished: HybirdNetworkAudio")
        return

    if train_config.TYPE == "HybirdNetwork":
        from network import HybirdNetwork
        from train import HybirdNetworkTrain

        model = HybirdNetwork()
        Train(model=model, trainer=HybirdNetworkTrain, config=train_config)
        logger.info("train finished: HybirdNetwork")
        return

    if train_config.TYPE == "OfflineRL":
        offline_config = train_config.OFFLINE
        sac_model, trainer, online_test = _build_offline_agent(config, offline_config)
        Train(
            model=sac_model,
            trainer=trainer,
            config=offline_config,
            online_test=online_test,
        )
        logger.info("train finished: OfflineRL")
        return

    raise ValueError(f"Unsupported TRAIN.TYPE: {train_config.TYPE}")


def main():
    task_config = config.TASK_CONFIG
    logger.info("run begin")
    logger.info(
        "task flags: collect=%s pt=%s eval=%s train=%s",
        task_config.COLLECT.OPEN,
        task_config.PT.OPEN,
        task_config.EVAL.OPEN,
        task_config.TRAIN.OPEN,
    )
    if task_config.COLLECT.OPEN:
        _run_collect(config, task_config.COLLECT)
    elif task_config.PT.OPEN:
        _run_pt(task_config.PT)
    elif task_config.EVAL.OPEN:
        _run_eval(config, task_config.EVAL)
    elif task_config.TRAIN.OPEN:
        _run_train(config, task_config.TRAIN)
    else:
        raise ValueError("No task is enabled. Please set one OPEN field to True.")
    logger.info("run finished")


config = get_config()

if __name__ == "__main__":
    run_name = "run"
    task_cfg = config.TASK_CONFIG
    if task_cfg.COLLECT.OPEN:
        run_name = f"collect_{task_cfg.COLLECT.TYPE}"
    elif task_cfg.PT.OPEN:
        run_name = f"pt_{task_cfg.PT.TYPE}"
    elif task_cfg.EVAL.OPEN:
        run_name = f"eval_{task_cfg.EVAL.TYPE}"
    elif task_cfg.TRAIN.OPEN:
        run_name = f"train_{task_cfg.TRAIN.TYPE}"

    log_path = setup_run_logger(base_dir="logs", run_name=run_name)
    logger.info("python run.py started, log_path=%s", log_path)
    exp_note = os.environ.get("EXP_NOTE", "").strip()
    journal_path = append_experiment_journal(
        run_name=run_name,
        note=exp_note,
        journal_path="logs/experiment_journal.md",
    )
    logger.info("experiment journal appended: %s", journal_path)
    try:
        main()
    except Exception:
        logger.exception("run failed with exception")
        raise
