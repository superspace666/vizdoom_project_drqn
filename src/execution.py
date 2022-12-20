import os
from logging import getLogger

import actions
from game import Game, GameFeaturesConfusionMatrix
from src.params_management import params_get
from src.params_management.params_get import parse_reward_values, get_model_class
from trainer import ReplayMemoryTrainer
import json
import torch
import pickle

logger = getLogger()


def evaluate_deathmatch(game, network, params, n_train_iter=None):
    logger.info('Evaluating the model...')
    game.statistics = {}

    n_features = params.n_features
    if n_features > 0:
        confusion = GameFeaturesConfusionMatrix(params.map_ids_test, n_features)

    # evaluate on every test map
    for map_id in params.map_ids_test:

        logger.info("Evaluating on map %i ..." % map_id)
        game.start(map_id=map_id, log_events=True,
                   manual_control=(params.manual_control and not params.human_player))
        game.randomize_textures(False)
        game.init_bots_health(100)
        network.reset()
        network.module.eval()

        n_iter = 0
        last_states = []

        while n_iter * params.frame_skip < params.eval_time * 35:
            n_iter += 1

            if game.is_player_dead():
                game.respawn_player()
                network.reset()

            while game.is_player_dead():
                logger.warning('Player %i is still dead after respawn.' %
                               params.player_rank)
                game.respawn_player()

            # observe the game state / select the next action
            game.observe_state(params, last_states)
            action = network.next_action(last_states)
            pred_features = network.pred_features

            # game features
            assert (pred_features is None) ^ n_features
            if n_features:
                assert pred_features.size() == (params.n_features,)
                pred_features = pred_features.data.cpu().numpy().ravel()
                confusion.update_predictions(pred_features,
                                             last_states[-1].features,
                                             game.map_id)

            sleep = 0.01 if params.evaluate else None
            game.make_action(action, params.frame_skip, sleep=sleep)

        # close the game
        game.close()

    # log the number of iterations and statistics
    logger.info("%i iterations" % n_iter)
    if n_features != 0:
        confusion.print_statistics()
    game.print_statistics()
    to_log = ['kills', 'deaths', 'suicides', 'frags', 'k/d']
    to_log = {k: game.statistics['all'][k] for k in to_log}
    if n_train_iter is not None:
        to_log['n_iter'] = n_train_iter
    logger.info("__log__:%s" % json.dumps(to_log))

    # evaluation score
    return game.statistics['all']['frags']


def main(parser, args, parameter_server=None):
    # register model and scenario parameters / parse parameters
    params_get.register_model_args(parser, args)  # 获得参数
    params_get.register_scenario_args(parser)  # 获得场景
    params = parser.parse_args(args)
    params.human_player = params.human_player and params.player_rank == 0

    # Game variables / Game features / feature maps
    params.game_variables = [('health', 101), ('sel_ammo', 301)]
    params_get.finalize_args(params)

    # Training / Evaluation parameters
    params.episode_time = None  # episode maximum duration (in seconds)
    params.eval_freq = 20000  # time (in iterations) between 2 evaluations
    params.eval_time = 900  # evaluation time (in seconds)

    # log experiment parameters
    with open(os.path.join(params.dump_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in dict(vars(params)).items()))

    # use only 1 CPU thread / set GPU ID if required
    set_num_threads(1)
    if params.gpu_id >= 0:
        torch.cuda.set_device(params.gpu_id)

    # Action builder
    action_builder = actions.ActionBuilder(params)

    # Initialize the game
    game = Game(
        scenario=params.wad,
        action_builder=action_builder,
        reward_values=parse_reward_values(params.reward_values),
        score_variable='USER2',
        freedoom=params.freedoom,
        # screen_resolution='RES_400X225',
        use_screen_buffer=params.use_screen_buffer,
        use_depth_buffer=params.use_depth_buffer,
        labels_mapping=params.labels_mapping,
        game_features=params.game_features,
        mode=('SPECTATOR' if params.human_player else 'PLAYER'),
        player_rank=params.player_rank,
        players_per_game=params.players_per_game,
        render_hud=params.render_hud,
        render_crosshair=params.render_crosshair,
        render_weapon=params.render_weapon,
        freelook=params.freelook,
        visible=params.visualize,
        n_bots=params.n_bots,
        use_scripted_marines=True
    )

    # Network initialization and optional reloading
    network = get_model_class(params.network_type)(params)
    if params.reload:
        logger.info('Reloading model from %s...' % params.reload)
        model_path = os.path.join(params.dump_path, params.reload)
        map_location = get_device_mapping(params.gpu_id)
        reloaded = torch.load(model_path, map_location=map_location)
        network.module.load_state_dict(reloaded)
    assert params.n_features == network.module.n_features

    # Parameter server (multi-agent training, self-play, etc.)
    if parameter_server:
        assert params.gpu_id == -1
        parameter_server.register_model(network.module)

    # Visualize only
    if params.evaluate:
        evaluate_deathmatch(game, network, params)
    else:
        logger.info('Starting experiment...')
        if params.network_type.startswith('dqn'):
            trainer_class = ReplayMemoryTrainer
        else:
            raise RuntimeError("unknown network type " + params.network_type)
        trainer_class(params, game, network, evaluate_deathmatch,
                      parameter_server=parameter_server).run()


def set_num_threads(n):
    assert n >= 1
    torch.set_num_threads(n)
    os.environ['MKL_NUM_THREADS'] = str(n)


def get_device_mapping(gpu_id):
    origins = ['cpu'] + ['cuda:%i' % i for i in range(8)]
    target = 'cpu' if gpu_id < 0 else 'cuda:%i' % gpu_id
    return {k: target for k in origins}
