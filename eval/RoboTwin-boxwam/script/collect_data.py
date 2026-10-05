import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None, collect_failure=False, failure_num=None,
         target_noise_std=0.0):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Allow the dataset root to be overridden via SAVE_PATH env var (e.g.
    # SAVE_PATH=/path/to/large/disk/robotwin) without editing the
    # task config yaml. Per-task / per-config subpath is appended below.
    _save_path_override = os.environ.get("SAVE_PATH")
    if _save_path_override:
        args["save_path"] = _save_path_override

    args['task_name'] = task_name
    args["collect_failure"] = collect_failure
    args["target_noise_std"] = float(target_noise_std)
    if collect_failure:
        if failure_num is None:
            failure_num = 2 * args["episode_num"]
        args["failure_num"] = failure_num
    if collect_failure or args["target_noise_std"] > 0.0:
        args["save_path"] = args["save_path"].rstrip("/") + "_fail"

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    collect_failure = args.get("collect_failure", False)
    target_count = args["failure_num"] if collect_failure else args["episode_num"]

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    if collect_failure:
        print("\033[91m[Failure-data mode] saving only failed episodes\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    user_target_noise_std = args.get("target_noise_std", 0.0)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True
        if user_target_noise_std > 0.0:
            print(f"\033[91m[Target-pose noise] std={user_target_noise_std} m on every planner target\033[0m")

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < target_count:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                episode_succeeded = TASK_ENV.plan_success and TASK_ENV.check_success()

                if collect_failure:
                    if episode_succeeded:
                        print(f"seed {epid} succeeded; skipping (failure-only mode)")
                    elif not TASK_ENV.plan_success:
                        print(f"seed {epid}: planner failed mid-chain; skipping (want full-execution failures)")
                    else:
                        # plan_success=True, check_success=False: full execution that didn't reach goal
                        print(f"capture failure episode {suc_num} (seed = {epid}, full execution, check_success=False)")
                        seed_list.append(epid)
                        TASK_ENV.save_traj_data(suc_num)
                        suc_num += 1
                else:
                    if episode_succeeded:
                        print(f"simulate data episode {suc_num} success! (seed = {epid})")
                        seed_list.append(epid)
                        TASK_ENV.save_traj_data(suc_num)
                        suc_num += 1
                    else:
                        print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                        fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True
        args["target_noise_std"] = 0.0

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, target_count):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            # A seed that was stable during Phase-1 planning can become unstable during Phase-2
            # replay (physics nondeterminism). Mirror the seed-collection loop's handling so a
            # single bad seed skips one episode instead of crashing the whole task mid-run
            # (which previously truncated the dataset, e.g. dump_bin_bigbin -> 475/500).
            try:
                TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

                traj_data = TASK_ENV.load_tran_data(episode_idx)
                args["left_joint_path"] = traj_data["left_joint_path"]
                args["right_joint_path"] = traj_data["right_joint_path"]
                TASK_ENV.set_path_lst(args)

                info_file_path = os.path.join(args["save_path"], "scene_info.json")

                if not os.path.exists(info_file_path):
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump({}, file, ensure_ascii=False)

                with open(info_file_path, "r", encoding="utf-8") as file:
                    info_db = json.load(file)

                info = TASK_ENV.play_once()
                info_db[f"episode_{episode_idx}"] = info

                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)

                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                if getattr(TASK_ENV, "FRAME_IDX", 0) == 0:
                    print(f"\033[93m[skip episode {episode_idx}] no frames captured during replay\033[0m")
                    continue
                TASK_ENV.merge_pkl_to_hdf5_video()
                TASK_ENV.save_obb2d()
                TASK_ENV.remove_data_cache()
                if not collect_failure and user_target_noise_std == 0.0:
                    assert TASK_ENV.check_success(), "Collect Error"
            except UnStableError as e:
                print(f"\033[93m[skip episode {episode_idx}] UnStableError on seed "
                      f"{seed_list[episode_idx]} during replay: {e}\033[0m")
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                time.sleep(0.3)
                continue

        if not collect_failure and user_target_noise_std == 0.0:
            command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
            os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--collect-failure", action="store_true",
                        help="Collect failure-only trajectories into ./<save_path>_fail/")
    parser.add_argument("--failure-num", type=int, default=None,
                        help="Number of failure episodes to collect (default: 2 * episode_num)")
    parser.add_argument("--target-noise-std", type=float, default=0.0,
                        help="Gaussian noise std (meters) added to each planner target end-effector position during Phase 1")
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config,
         collect_failure=parser.collect_failure,
         failure_num=parser.failure_num,
         target_noise_std=parser.target_noise_std)
