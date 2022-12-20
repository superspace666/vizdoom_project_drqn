import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(string):
    if string.lower() in FALSY_STRINGS:  # string.lower():返回string的全小写
        return False
    elif string.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. "
                                         "use 0 or 1")


logger = getLogger()


def value_loss(delta):

    assert delta >= 0
    if delta == 0:
        # MSE Loss
        return nn.MSELoss()
    elif delta == 1:
        # Smooth L1 Loss
        return nn.SmoothL1Loss()
    else:
        # Huber Loss
        def loss_fn(input, target):
            diff = (input - target).abs()
            diff_delta = diff.cmin(delta)
            loss = diff_delta * (diff - diff_delta / 2)
            return loss.mean()

        return loss_fn


class DQNModuleBase(nn.Module):

    def __init__(self, params):
        super(DQNModuleBase, self).__init__()

        # CNN
        build_CNN_network(self, params)
        self.output_dim = self.conv_output_dim

        # game variables
        build_game_variables_network(self, params)
        if self.n_variables:
            self.output_dim += sum(params.variable_dim)

        # dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

        # game features
        build_game_features_network(self, params)

        # Q
        self.proj_action_scores = nn.Linear(params.hidden_dim, self.n_actions)
        self.dueling_network = params.dueling_network
        self.proj_state_values = nn.Linear(params.hidden_dim, 1)

        logger.info('Conv layer output dim : %i' % self.conv_output_dim)
        logger.info('Hidden layer input dim: %i' % self.output_dim)

    def base_forward(self, x_screens, x_variables):

        batch_size = x_screens.size(0)

        # convolution
        x_screens = x_screens / 255.
        conv_output = self.conv(x_screens).view(batch_size, -1)

        # game variables
        if self.n_variables:
            embeddings = [self.game_variable_embeddings[i](x_variables[i])
                          for i in range(self.n_variables)]

        # game features
        if self.n_features:
            output_gf = self.proj_game_features(conv_output)
        else:
            output_gf = None

        # create state input
        if self.n_variables:
            output = torch.cat([conv_output] + embeddings, 1)
        else:
            output = conv_output

        # dropout
        if self.dropout:
            output = self.dropout_layer(output)

        return output, output_gf

    def head_forward(self, state_input):
        if self.dueling_network:
            a = self.proj_action_scores(state_input)  # advantage branch
            v = self.proj_state_values(state_input)  # state value branch
            a -= a.mean(1, keepdim=True).expand(a.size())
            return v.expand(a.size()) + a
        else:
            return self.proj_action_scores(state_input)


def build_CNN_network(module, params):
    # model parameters
    module.hidden_dim = params.hidden_dim
    module.dropout = params.dropout
    module.n_actions = params.n_actions

    # screen input format - for RNN, we only take one frame at each time step
    if hasattr(params, 'recurrence') and params.recurrence != '':
        in_channels = params.n_fm
    else:
        in_channels = params.n_fm * params.hist_size
    height = params.height
    width = params.width
    logger.info('Input shape: %s' % str((params.n_fm, height, width)))

    # convolutional layers
    module.conv = nn.Sequential(*filter(bool, [
        nn.Conv2d(in_channels, 32, (8, 8), stride=(4, 4)),
        # None if not params.use_bn else nn.BatchNorm2d(32),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
        # None if not params.use_bn else nn.BatchNorm2d(64),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    ]))

    x = Variable(torch.FloatTensor(1, in_channels, height, width).zero_())
    module.conv_output_dim = module.conv(x).nelement()


def build_game_variables_network(module, params):
    module.game_variables = params.game_variables
    module.n_variables = params.n_variables
    module.game_variable_embeddings = []
    for i, (name, n_values) in enumerate(params.game_variables):
        embeddings = BucketedEmbedding(params.bucket_size[i], n_values,
                                       params.variable_dim[i])
        setattr(module, '%s_emb' % name, embeddings)
        module.game_variable_embeddings.append(embeddings)


def build_game_features_network(module, params):
    module.game_features = params.game_features
    if module.game_features:
        module.n_features = module.game_features.count(',') + 1
        module.proj_game_features = nn.Sequential(
            nn.Dropout(module.dropout),
            nn.Linear(module.conv_output_dim, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(module.dropout),
            nn.Linear(params.hidden_dim, module.n_features),
            nn.Sigmoid()
        )
    else:
        module.n_features = 0


class BucketedEmbedding(nn.Embedding):

    def __init__(self, bucket_size, num_embeddings, *args, **kwargs):
        self.bucket_size = bucket_size
        real_num_embeddings = (num_embeddings + bucket_size - 1) // bucket_size
        super(BucketedEmbedding, self).__init__(real_num_embeddings, *args, **kwargs)

    def forward(self, indices):
        indices_ = torch.LongTensor(
            (indices.div(self.bucket_size)).cpu().numpy()).cuda()  # torch.cuda.LongTensor(indices)
        return super(BucketedEmbedding, self).forward(indices_)


class DQNModuleFeedforward(DQNModuleBase):
    def __init__(self, params):
        super(DQNModuleFeedforward, self).__init__(params)

        self.feedforward = nn.Sequential(
            nn.Linear(self.output_dim, params.hidden_dim),
            nn.Sigmoid())

    def forward(self, x_screens, x_variables):

        batch_size = x_screens.size(0)
        assert x_screens.ndimension() == 4
        assert len(x_variables) == self.n_variables
        assert all(x.ndimension() == 1 and x.size(0) == batch_size
                   for x in x_variables)

        # state input (screen / depth / labels buffer + variables)
        state_input, output_gf = self.base_forward(x_screens, x_variables)

        # apply the feed forward middle
        state_input = self.feedforward(state_input)

        # apply the head to feed forward result
        output_sc = self.head_forward(state_input)

        return output_sc, output_gf


class DQN(object):

    def __init__(self, params):
        # network parameters
        self.params = params
        self.screen_shape = (params.n_fm, params.height, params.width)
        self.hist_size = params.hist_size
        self.n_variables = params.n_variables
        self.n_features = params.n_features

        # main module + loss functions
        self.module = self.DQNModuleClass(params)
        self.loss_fn_sc = value_loss(params.clip_delta)
        self.loss_fn_gf = nn.BCELoss()

        # cuda
        self.cuda = params.gpu_id >= 0
        if self.cuda:
            self.module.cuda()

    def get_var(self, x):

        x = Variable(x)
        return x.cuda() if self.cuda else x

    def reset(self):
        pass

    def new_loss_history(self):
        return dict(dqn_loss=[], gf_loss=[])

    def log_loss(self, loss_history):
        logger.info('DQN loss: %.5f' % np.mean(loss_history['dqn_loss']))
        if self.n_features > 0:
            logger.info('Game features loss: %.5f' %
                        np.mean(loss_history['gf_loss']))

    def prepare_f_eval_args(self, last_states):

        screens = np.float32([s.screen for s in last_states])
        screens = self.get_var(torch.FloatTensor(screens))
        assert screens.size() == (self.hist_size,) + self.screen_shape

        if self.n_variables:
            variables = np.int64([s.variables for s in last_states])
            variables = self.get_var(torch.LongTensor(variables))
            assert variables.size() == (self.hist_size, self.n_variables)
        else:
            variables = None

        return screens, variables

    def prepare_f_train_args(self, screens, variables, features,
                             actions, rewards, isfinal):

        # convert tensors to torch Variables
        screens = self.get_var(torch.FloatTensor(np.float32(screens).copy()))
        if self.n_variables:
            variables = self.get_var(torch.LongTensor(np.int64(variables).copy()))
        if self.n_features:
            features = self.get_var(torch.LongTensor(np.int64(features).copy()))
        rewards = self.get_var(torch.FloatTensor(np.float32(rewards).copy()))
        isfinal = self.get_var(torch.FloatTensor(np.float32(isfinal).copy()))

        recurrence = self.params.recurrence
        batch_size = self.params.batch_size
        n_updates = 1 if recurrence == '' else self.params.n_rec_updates
        seq_len = self.hist_size + n_updates

        # check tensors sizes
        assert screens.size() == (batch_size, seq_len) + self.screen_shape
        if self.n_variables:
            assert variables.size() == (batch_size, seq_len, self.n_variables)
        if self.n_features:
            assert features.size() == (batch_size, seq_len, self.n_features)
        assert actions.shape == (batch_size, seq_len - 1)
        assert rewards.size() == (batch_size, seq_len - 1)
        assert isfinal.size() == (batch_size, seq_len - 1)

        return screens, variables, features, actions, rewards, isfinal

    def register_loss(self, loss_history, loss_sc, loss_gf):
        loss_history['dqn_loss'].append(loss_sc.item())
        loss_history['gf_loss'].append(loss_gf.item()
                                       if self.n_features else 0)

    def next_action(self, last_states, save_graph=False):
        scores, pred_features = self.f_eval(last_states)
        if self.params.network_type == 'dqn_ff':
            assert scores.size() == (1, self.module.n_actions)
            scores = scores[0]
            if pred_features is not None:
                assert pred_features.size() == (1, self.module.n_features)
                pred_features = pred_features[0]
        else:
            assert self.params.network_type == 'dqn_rnn'
            seq_len = 1 if self.params.remember else self.params.hist_size
            assert scores.size() == (1, seq_len, self.module.n_actions)
            scores = scores[0, -1]
            if pred_features is not None:
                assert pred_features.size() == (1, seq_len, self.module.n_features)
                pred_features = pred_features[0, -1]
        action_id = scores.data.max(0)[1][0]
        self.pred_features = pred_features
        return action_id

    def register_args(parser):
        # batch size / replay memory size
        parser.add_argument("--batch_size", type=int, default=32,
                            help="Batch size")
        parser.add_argument("--replay_memory_size", type=int, default=100000,
                            help="Replay memory size")

        # epsilon decay
        parser.add_argument("--start_decay", type=int, default=0,
                            help="Learning step when the epsilon decay starts")
        parser.add_argument("--stop_decay", type=int, default=100000,
                            help="Learning step when the epsilon decay stops")
        parser.add_argument("--final_decay", type=float, default=0.1,
                            help="Epsilon value after decay")

        # discount factor / dueling architecture
        parser.add_argument("--gamma", type=float, default=0.99,
                            help="Gamma")
        parser.add_argument("--dueling_network", type=bool_flag, default=False,
                            help="Use a dueling network architecture")

        # recurrence type
        parser.add_argument("--recurrence", type=str, default='',
                            help="Recurrent neural network (RNN / GRU / LSTM)")

    def validate_params(params):
        assert 0 <= params.start_decay <= params.stop_decay
        assert 0 <= params.final_decay <= 1
        assert params.replay_memory_size >= 1000


class DQNFeedforward(DQN):
    DQNModuleClass = DQNModuleFeedforward

    def f_eval(self, last_states):

        screens, variables = self.prepare_f_eval_args(last_states)

        return self.module(
            screens.view(1, -1, *self.screen_shape[1:]),
            [variables[-1, i] for i in range(self.params.n_variables)]
        )

    def f_train(self, screens, variables, features, actions, rewards, isfinal,
                loss_history=None):

        screens, variables, features, actions, rewards, isfinal = \
            self.prepare_f_train_args(screens, variables, features,
                                      actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + 1

        screens = screens.view(batch_size, seq_len * self.params.n_fm,
                               *self.screen_shape[1:])

        output_sc1, output_gf1 = self.module(
            screens[:, :-self.params.n_fm, :, :],
            [variables[:, -2, i] for i in range(self.params.n_variables)]
        )
        output_sc2, output_gf2 = self.module(
            screens[:, self.params.n_fm:, :, :],
            [variables[:, -1, i] for i in range(self.params.n_variables)]
        )

        # compute scores
        mask = torch.ByteTensor(output_sc1.size()).fill_(0)
        for i in range(batch_size):
            mask[i, int(actions[i, -1])] = 1
        scores1 = output_sc1.masked_select(self.get_var(mask))
        scores2 = rewards[:, -1] + (
                self.params.gamma * output_sc2.max(1)[0] * (1 - isfinal[:, -1])
        )

        # dqn loss
        loss_sc = self.loss_fn_sc(scores1, Variable(scores2.data))

        # game features loss
        loss_gf = 0
        if self.n_features:
            loss_gf += self.loss_fn_gf(output_gf1, features[:, -2].float())
            loss_gf += self.loss_fn_gf(output_gf2, features[:, -1].float())

        self.register_loss(loss_history, loss_sc, loss_gf)

        return loss_sc, loss_gf

    def validate_params(params):
        DQN.validate_params(params)
        assert params.recurrence == ''


class DQNModuleRecurrent(DQNModuleBase):

    def __init__(self, params):
        super(DQNModuleRecurrent, self).__init__(params)

        self.rnn = nn.LSTM(self.output_dim, params.hidden_dim,
                           num_layers=params.n_rec_layers,
                           dropout=params.dropout,
                           batch_first=True)

    def forward(self, x_screens, x_variables, prev_state):
        batch_size = x_screens.size(0)
        seq_len = x_screens.size(1)

        assert x_screens.ndimension() == 5
        assert len(x_variables) == self.n_variables
        assert all(x.ndimension() == 2 and x.size(0) == batch_size and
                   x.size(1) == seq_len for x in x_variables)

        # We're doing a batched forward through the network base
        # Flattening seq_len into batch_size ensures that it will be applied
        # to all timesteps independently.
        state_input, output_gf = self.base_forward(
            x_screens.view(batch_size * seq_len, *x_screens.size()[2:]),
            [v.contiguous().view(batch_size * seq_len) for v in x_variables]
        )

        # unflatten the input and apply the RNN
        rnn_input = state_input.view(batch_size, seq_len, self.output_dim)
        rnn_output, next_state = self.rnn(rnn_input, prev_state)
        rnn_output = rnn_output.contiguous()

        # apply the head to RNN hidden states (simulating larger batch again)
        output_sc = self.head_forward(rnn_output.view(-1, self.hidden_dim))

        # unflatten scores and game features
        output_sc = output_sc.view(batch_size, seq_len, output_sc.size(1))
        if self.n_features:
            output_gf = output_gf.view(batch_size, seq_len, self.n_features)

        return output_sc, output_gf, next_state


class DQNRecurrent(DQN):
    DQNModuleClass = DQNModuleRecurrent

    def __init__(self, params):
        super(DQNRecurrent, self).__init__(params)
        h_0 = torch.FloatTensor(params.n_rec_layers, params.batch_size,
                                params.hidden_dim).zero_()
        self.init_state_t = self.get_var(h_0)
        self.init_state_e = Variable(self.init_state_t[:, :1, :].data.clone(), volatile=True)
        if params.recurrence == 'lstm':
            self.init_state_t = (self.init_state_t, self.init_state_t)
            self.init_state_e = (self.init_state_e, self.init_state_e)
        self.reset()

    def reset(self):
        # prev_state is only used for evaluation, so has a batch size of 1
        self.prev_state = self.init_state_e

    def f_eval(self, last_states):

        screens, variables = self.prepare_f_eval_args(last_states)

        # if we remember the whole sequence, only feed the last frame
        if self.params.remember:
            output = self.module(
                screens[-1:].view(1, 1, *self.screen_shape),
                [variables[-1:, i].view(1, 1)
                 for i in range(self.params.n_variables)],
                prev_state=self.prev_state
            )
            # save the hidden state if we want to remember the whole sequence
            self.prev_state = output[-1]
        # otherwise, feed the last `hist_size` ones
        else:
            output = self.module(
                screens.view(1, self.hist_size, *self.screen_shape),
                [variables[:, i].contiguous().view(1, self.hist_size)
                 for i in range(self.params.n_variables)],
                prev_state=self.prev_state
            )

        # do not return the recurrent state
        return output[:-1]

    def f_train(self, screens, variables, features, actions, rewards, isfinal,
                loss_history=None):

        screens, variables, features, actions, rewards, isfinal = \
            self.prepare_f_train_args(screens, variables, features,
                                      actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + self.params.n_rec_updates

        output_sc, output_gf, _ = self.module(
            screens,
            [variables[:, :, i] for i in range(self.params.n_variables)],
            prev_state=self.init_state_t
        )

        # compute scores
        mask = torch.ByteTensor(output_sc.size()).fill_(0)
        for i in range(batch_size):
            for j in range(seq_len - 1):
                mask[i, j, int(actions[i, j])] = 1
        scores1 = output_sc.masked_select(self.get_var(mask))
        scores2 = rewards + (
                self.params.gamma * output_sc[:, 1:, :].max(2)[0] * (1 - isfinal)
        )

        # dqn loss
        loss_sc = self.loss_fn_sc(
            scores1.view(batch_size, -1)[:, -self.params.n_rec_updates:],
            Variable(scores2.data[:, -self.params.n_rec_updates:])
        )

        # game features loss
        if self.n_features:
            loss_gf = self.loss_fn_gf(output_gf, features.float())
        else:
            loss_gf = 0

        self.register_loss(loss_history, loss_sc, loss_gf)

        return loss_sc, loss_gf

    def register_args(parser):
        DQN.register_args(parser)
        parser.add_argument("--n_rec_updates", type=int, default=1,
                            help="Number of updates to perform")
        parser.add_argument("--n_rec_layers", type=int, default=1,
                            help="Number of recurrent layers")
        parser.add_argument("--remember", type=bool_flag, default=True,
                            help="Remember the whole sequence")

    def validate_params(params):
        DQN.validate_params(params)
        assert params.recurrence in ['rnn', 'gru', 'lstm']
        assert params.n_rec_updates >= 1
        assert params.n_rec_layers >= 1
