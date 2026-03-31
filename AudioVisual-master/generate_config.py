import os
import shutil
from ruamel.yaml import YAML
from itertools import product

yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

baselines = ["av_nav", "av_wan"]
envs = ["replica", "mp3d"]

for baseline, env in product(baselines, envs):
    if baseline == "av_nav":
        val_config_file = "audiogoal_val.yaml"
    else:
        val_config_file = "audiogoal_val_multiple.yaml"
    with open(os.path.join("configs", "audionav", baseline, env, val_config_file), "r") as f:
        config = yaml.load(f)
    config["DATASET"]["SPLIT"] = "val_multiple"
    val_config_file = val_config_file.split(".")[0] + "_heard.yaml"
    with open(os.path.join("configs", "audionav", baseline, env, val_config_file), "w") as f:
        yaml.dump(config, f)

    if baseline == "av_nav":
        test_config_file = "audiogoal_test.yaml"
    else:
        test_config_file = "audiogoal_test_multiple.yaml"
    with open(os.path.join("configs", "audionav", baseline, env, test_config_file), "r") as f:
        config = yaml.load(f)
    config["DATASET"]["SPLIT"] = "test_multiple"
    test_config_file = test_config_file.split(".")[0] + "_heard.yaml"
    with open(os.path.join("configs", "audionav", baseline, env, test_config_file), "w") as f:
        yaml.dump(config, f)

    with open(os.path.join("ss_baselines", baseline, "config", "audionav", env, "val", "val_template.yaml"), "r") as f:
        config = yaml.load(f)
    config["BASE_TASK_CONFIG_PATH"] = os.path.join("configs", "audionav", baseline, env, val_config_file)
    config["EVAL"]["SPLIT"] = "val_multiple"
    with open(os.path.join("ss_baselines", baseline, "config", "audionav", env, "val", "val_template_heard.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(os.path.join("ss_baselines", baseline, "config", "audionav", env, "test", "test_template.yaml"), "r") as f:
        config = yaml.load(f)
    config["BASE_TASK_CONFIG_PATH"] = os.path.join("configs", "audionav", baseline, env, test_config_file)
    config["EVAL"]["SPLIT"] = "test_multiple"
    with open(os.path.join("ss_baselines", baseline, "config", "audionav", env, "test", "test_template_heard.yaml"), "w") as f:
        yaml.dump(config, f)
