import numpy as np
import torch

from boxwam.data.normalization import ActionStats
from boxwam.model.boxwam import BoxWAM
from boxwam.training.checkpoint import load_checkpoint, load_model_weights, save_checkpoint, stats_from_dict


def test_save_and_reload_round_trip(tmp_path, tiny_model):
    stats = ActionStats(
        action_q01=np.zeros(4, np.float32),
        action_q99=np.ones(4, np.float32),
        state_q01=np.zeros(3, np.float32),
        state_q99=np.ones(3, np.float32),
        gripper_dims=(3,),
    )
    save_checkpoint(tmp_path / "ck.pt", model=tiny_model, config={"model": {}}, stats=stats, epoch=3)
    checkpoint = load_checkpoint(tmp_path / "ck.pt")
    assert checkpoint["epoch"] == 3 and stats_from_dict(checkpoint["stats"]).gripper_dims == (3,)
    assert isinstance(checkpoint["stats"]["action_q01"], list), "plain lists keep checkpoints portable"
    clone = BoxWAM(
        encoder_dim=32,
        embed_dim=48,
        depth=2,
        num_heads=2,
        image_size=32,
        patch_size=16,
        num_cameras=1,
        state_dim=3,
        action_dim=4,
        horizon=3,
        box_dim=16,
    )
    load_model_weights(clone, checkpoint)
    for a, b in zip(tiny_model.parameters(), clone.parameters()):
        assert torch.equal(a, b)
