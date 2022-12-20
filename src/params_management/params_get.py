import importlib
import json
import os
import argparse

from src.models.DQN_base import DQNFeedforward, DQNRecurrent

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}
models = {
    'dqn_ff': DQNFeedforward,
    'dqn_rnn': DQNRecurrent
}


def bool_flag(string):
    if string.lower() in FALSY_STRINGS:  # string.lower():返回string的全小写
        return False
    elif string.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. "
                                         "use 0 or 1")


def get_model_class(model_type):
    cls = models.get(model_type)
    if cls is None:
        raise RuntimeError(("unknown model type: '%s'. supported values "
                            "are: %s") % (model_type, ', '.join(models.keys())))
    return cls


def register_model_args(parser, args):
    # network type
    parser.add_argument("--network_type", type=str, default='dqn_ff',
                        help="Network type (dqn_ff / dqn_rnn)")
    parser.add_argument("--use_bn", type=bool_flag, default=False,
                        help="Use batch normalization in CNN network")

    # model parameters
    params, _ = parser.parse_known_args(args)
    network_class = get_model_class(params.network_type)
    network_class.register_args(parser)
    network_class.validate_params(parser.parse_known_args(args)[0])

    # parameters common to all models
    parser.add_argument("--clip_delta", type=float, default=1.0,
                        help="Clip delta")
    parser.add_argument("--variable_dim", type=str, default='32',
                        help="Game variables embeddings dimension")
    parser.add_argument("--bucket_size", type=str, default='1',
                        help="Bucket size for game variables")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer dimension")
    parser.add_argument("--update_frequency", type=int, default=4,
                        help="Update frequency (1 for every time)")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="Dropout")
    parser.add_argument("--optimizer", type=str, default="rmsprop,lr=0.0002",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")

    # check common parameters
    params, _ = parser.parse_known_args(args)
    assert params.clip_delta >= 0
    assert params.update_frequency >= 1
    assert 0 <= params.dropout < 1


def register_scenario_args(parser):
    parser.add_argument("--wad", type=str, default="",
                        help="WAD scenario filename")
    parser.add_argument("--n_bots", type=int, default=8,
                        help="Number of ACS bots in the game")
    parser.add_argument("--reward_values", type=str, default="",
                        help="reward_values")
    parser.add_argument("--randomize_textures", type=bool_flag, default=False,
                        help="Randomize textures during training")
    parser.add_argument("--init_bots_health", type=int, default=100,
                        help="Initial bots health during training")


def parse_game_features(s):
    game_features = ['target', 'enemy', 'health', 'weapon', 'ammo']
    split = list(filter(None, s.split(',')))
    assert all(x in game_features for x in split)
    return [x in split for x in game_features]


def parse_labels_mapping(s):
    if len(s) > 0:
        split = [[int(y) for y in x.split('+')] for x in s.split(';')]
        elements = sum(split, [])
        assert all(x in range(4) for x in elements)
        assert len(elements) == len(set(elements))
        labels_mapping = []
        for i in range(4):
            found = False
            for j, l in enumerate(split):
                if i in l:
                    assert not found
                    found = True
                    labels_mapping.append(j)
            if not found:
                labels_mapping.append(None)
        assert len(labels_mapping) == 4
    else:
        labels_mapping = None
    return labels_mapping


def get_n_feature_maps(params):
    n = 0
    if params.use_screen_buffer:
        n += 1 if params.gray else 3
    if params.use_depth_buffer:
        n += 1
    labels_mapping = parse_labels_mapping(params.labels_mapping)
    if labels_mapping is not None:
        n += len(set([x for x in labels_mapping if x is not None]))
    return n


def bcast_json_list(param, length):

    obj = json.loads(param)
    if isinstance(obj, list):
        assert len(obj) == length
        return obj
    else:
        assert isinstance(obj, int)
        return [obj] * length


def finalize_args(params):

    params.n_variables = len(params.game_variables)
    params.n_features = sum(parse_game_features(params.game_features))
    params.n_fm = get_n_feature_maps(params)

    params.variable_dim = bcast_json_list(params.variable_dim, params.n_variables)
    params.bucket_size = bcast_json_list(params.bucket_size, params.n_variables)

    if not hasattr(params, 'use_continuous'):  # 检测params里是否有use_continuous
        params.use_continuous = False


def parse_reward_values(reward_values):

    values = reward_values.split(',')
    reward_values = {}
    for x in values:
        if x == '':
            continue
        split = x.split('=')
        assert len(split) == 2
        reward_values[split[0]] = float(split[1])
    return reward_values


def map_ids_flag(string):

    ids = string.split(',')
    assert len(ids) >= 1 and all(x.isdigit() for x in ids)  # all()用于判断是否所有元素都为真，.isdiigit()判断内容是否是数字
    ids = sorted([int(x) for x in ids])  # sort()排序
    assert all(x >= 1 for x in ids) and len(ids) == len(set(ids))  # 这里是判断ids是否有重复#set() 函数创建一个无序不重复元素集
    return ids


def parse_game_args(args):

    parser = argparse.ArgumentParser(description='Doom parameters')

    # Doom scenario / map ID
    parser.add_argument("--scenario", type=str, default="deathmatch",
                        help="Doom scenario")
    parser.add_argument("--map_ids_train", type=map_ids_flag, default=map_ids_flag("1"),
                        help="Train map IDs")
    parser.add_argument("--map_ids_test", type=map_ids_flag, default=map_ids_flag("1"),
                        help="Test map IDs")

    # general game options (freedoom, screen resolution, available buffers,
    # game features, things to render, history size, frame skip, etc)
    parser.add_argument("--freedoom", type=bool_flag, default=True,
                        help="Use freedoom2.wad (as opposed to DOOM2.wad)")
    parser.add_argument("--height", type=int, default=60,
                        help="Image height")
    parser.add_argument("--width", type=int, default=108,
                        help="Image width")
    parser.add_argument("--gray", type=bool_flag, default=False,
                        help="Use grayscale")
    parser.add_argument("--use_screen_buffer", type=bool_flag, default=True,
                        help="Use the screen buffer")
    parser.add_argument("--use_depth_buffer", type=bool_flag, default=False,
                        help="Use the depth buffer")
    parser.add_argument("--labels_mapping", type=str, default='',
                        help="Map labels to different feature maps")
    parser.add_argument("--game_features", type=str, default='',
                        help="Game features")
    parser.add_argument("--render_hud", type=bool_flag, default=False,
                        help="Render HUD")
    parser.add_argument("--render_crosshair", type=bool_flag, default=True,
                        help="Render crosshair")
    parser.add_argument("--render_weapon", type=bool_flag, default=True,
                        help="Render weapon")
    parser.add_argument("--hist_size", type=int, default=4,
                        help="History size")
    parser.add_argument("--frame_skip", type=int, default=4,
                        help="Number of frames to skip")

    # Available actions
    # combination of actions the agent is allowed to do.
    # this is for non-continuous mode only, and is ignored in continuous mode
    parser.add_argument("--action_combinations", type=str,
                        default='move_fb+turn_lr+move_lr+attack',
                        help="Allowed combinations of actions")
    # freelook: allow the agent to look up and down
    parser.add_argument("--freelook", type=bool_flag, default=False,
                        help="Enable freelook (look up / look down)")
    # speed and crouch buttons: in non-continuous mode, the network can not
    # have control on these buttons, and they must be set to always 'on' or
    # 'off'. In continuous mode, the network can manually control crouch and
    # speed.
    # manual_control makes the agent turn about (180 degrees turn) if it keeps
    # repeating the same action (if it is stuck in one corner, for instance)
    parser.add_argument("--speed", type=str, default='off',
                        help="Crouch: on / off / manual")
    parser.add_argument("--crouch", type=str, default='off',
                        help="Crouch: on / off / manual")
    parser.add_argument("--manual_control", type=bool_flag, default=False,
                        help="Manual control to avoid action repetitions")

    # number of players / games
    parser.add_argument("--players_per_game", type=int, default=1,
                        help="Number of players per game")
    parser.add_argument("--player_rank", type=int, default=0,
                        help="Player rank")

    # miscellaneous
    parser.add_argument("--dump_path", type=str, default=".",
                        help="Folder to store the models / parameters.")
    parser.add_argument("--visualize", type=bool_flag, default=False,
                        help="Visualize")
    parser.add_argument("--evaluate", type=int, default=0,
                        help="Fast evaluation of the model")
    parser.add_argument("--human_player", type=bool_flag, default=False,
                        help="Human player (SPECTATOR mode)")
    parser.add_argument("--reload", type=str, default="",
                        help="Reload previous model")
    parser.add_argument("--dump_freq", type=int, default=0,
                        help="Dump every X iterations (0 to disable)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--log_frequency", type=int, default=100,
                        help="Log frequency (in seconds)")

    # Parse known arguments
    params, _ = parser.parse_known_args(args)

    # check parameters
    assert len(params.dump_path) > 0 and os.path.isdir(params.dump_path)
    assert len(params.scenario) > 0
    assert params.freelook ^ ('look_ud' not in params.action_combinations)
    assert {params.speed, params.crouch}.issubset(['on', 'off', 'manual'])
    assert not params.visualize or params.evaluate
    assert not params.human_player or params.evaluate and params.visualize
    assert not params.evaluate or params.reload
    assert not params.reload or os.path.isfile(params.reload)

    # run scenario game
    module = importlib.import_module('...execution', package=__name__)
    module.main(parser, args)
