import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../index'))

from models.rqvae import RQVAE


class TigerRQVAE(RQVAE):

    def __init__(self, **kwargs):
        num_layers = len(kwargs.get('num_emb_list', [256, 256, 256, 256]))
        kwargs['sk_epsilons'] = [0.0] * num_layers
        super().__init__(**kwargs)
        self.use_ot_loss = False
        self.ot_loss_weight = 0.0
