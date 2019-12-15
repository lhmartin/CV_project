#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sslime.models.heads.evaluation_mlp import Eval_MLP
from sslime.models.heads.mlp import MLP
from sslime.models.heads.jig_head import JIG_HEAD

HEADS = {"eval_mlp": Eval_MLP, "mlp": MLP, "jig_head" : JIG_HEAD}
