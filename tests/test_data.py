import numpy as np
import torch
from conftest import write_episode

from boxwam.data.augmentation import AugmentationConfig, _transform_boxes, augment
from boxwam.data.episodes import EpisodeDataset, find_episodes, split_episodes
from boxwam.data.loader import BalancedEpisodeSampler
from boxwam.data.normalization import ActionStats, denormalize_boxes, normalize_boxes


def make_stats(action_dim=4, state_dim=3, gripper=(3,)):
    return ActionStats(
        action_q01=np.full(action_dim, -2.0, dtype=np.float32),
        action_q99=np.full(action_dim, 2.0, dtype=np.float32),
        state_q01=np.full(state_dim, -1.0, dtype=np.float32),
        state_q99=np.full(state_dim, 3.0, dtype=np.float32),
        gripper_dims=gripper,
    )


def test_action_normalisation_round_trip():
    stats = make_stats()
    state = torch.tensor([[0.5, 1.0, 2.0]])
    actions = torch.tensor([[[0.6, 1.1, 1.9, 0.8], [0.7, 1.2, 1.8, 0.0]]])
    normalized = stats.normalize_actions(actions, state)
    assert torch.allclose(normalized[..., 3], actions[..., 3]), "gripper stays absolute"
    assert torch.allclose(normalized[0, 0, 0], torch.tensor((0.1 + 2.0) / 4.0 * 2 - 1), atol=1e-4)
    torch.testing.assert_close(stats.denormalize_actions(normalized, state), actions, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        stats.normalize_state(torch.tensor([[-1.0, 1.0, 3.0]])), torch.tensor([[-1.0, 0.0, 1.0]]), atol=1e-4
    )


def test_box_normalisation():
    boxes = torch.tensor([[[0.0, 112.0, 224.0, 250.0]]])
    normalized = normalize_boxes(boxes, 224)
    assert torch.allclose(normalized, torch.tensor([[[-1.0, 0.0, 1.0, 1.1]]]), atol=1e-6), "clamped 5% outside"
    assert torch.allclose(denormalize_boxes(normalized) * 224, torch.tensor([[[0.0, 112.0, 224.0, 235.2]]]), atol=1e-3)


def test_identity_augmentation_keeps_pixels_and_boxes():
    images = [torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8)]
    boxes = torch.rand(2, 3, 16) * 31
    out, out_boxes = augment(images, boxes, AugmentationConfig(crop_size=32))
    assert out[0].shape == (2, 3, 32, 32)
    expected = images[0].permute(0, 3, 1, 2).float() / 255.0
    expected = (expected - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    torch.testing.assert_close(out[0], expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_boxes, boxes, atol=1e-4, rtol=1e-5)


def test_flip_mirrors_boxes():
    params = {
        "top": torch.zeros(1),
        "left": torch.zeros(1),
        "crop_h": torch.tensor([32.0]),
        "crop_w": torch.tensor([32.0]),
        "angle": torch.zeros(1),
        "flip": torch.tensor([True]),
    }
    boxes = torch.tensor([[[3.0, 5.0]]])
    flipped = _transform_boxes(boxes, params, 32)
    assert torch.allclose(flipped, torch.tensor([[[31.0 - 3.0, 5.0]]]))


def test_episode_dataset_windows(tmp_path):
    episode = write_episode(tmp_path, "ep0", length=12)
    dataset = EpisodeDataset([episode], cameras=["cam"], horizon=3, load_boxes=True)
    sample = dataset[0]
    assert sample["images"][0].shape == (32, 32, 3) and sample["images"][0].dtype == torch.uint8
    assert sample["state"].shape == (3,) and sample["actions"].shape == (3, 4) and sample["boxes"].shape == (3, 16)
    state = np.load(episode / "state.npy")
    start = int(np.argmin(np.abs(state - sample["state"].numpy()).sum(1)))
    action = np.load(episode / "action.npy")
    boxes = np.load(episode / "proj_3d_obb.npy").reshape(12, -1)
    np.testing.assert_allclose(sample["actions"].numpy(), action[start : start + 3])
    np.testing.assert_allclose(sample["boxes"].numpy(), boxes[start + 1 : start + 4], rtol=1e-6)


def test_valid_starts_and_short_episodes(tmp_path):
    fixed = write_episode(tmp_path, "fixed", length=12, valid_starts=[4])
    dataset = EpisodeDataset([fixed], cameras=["cam"], horizon=3, use_valid_starts=True)
    state = np.load(fixed / "state.npy")
    for _ in range(5):
        np.testing.assert_allclose(dataset[0]["state"].numpy(), state[4])
    short = write_episode(tmp_path, "short", length=2)
    sample = EpisodeDataset([short], cameras=["cam"], horizon=3)[0]
    action = np.load(short / "action.npy")
    np.testing.assert_allclose(sample["actions"].numpy(), action[[0, 1, 1]])


def test_find_and_split_episodes(tmp_path):
    for i in range(10):
        write_episode(tmp_path / "root" / ("task_a" if i < 6 else "task_b"), f"ep{i}", length=5, boxes=None)
    episodes = find_episodes(tmp_path / "root")
    assert len(episodes) == 10 and len(find_episodes(tmp_path / "root", task="task_b")) == 4
    listing = tmp_path / "episodes.txt"
    listing.write_text("\n".join(str(path) for path in episodes) + "\n")
    assert find_episodes(listing) == episodes
    train, val = split_episodes(episodes, 0.2, seed=1)
    assert len(val) == 2 and len(train) == 8 and not set(train) & set(val)
    assert split_episodes(episodes, 0.2, seed=1) == (train, val), "deterministic"


def test_balanced_sampler_covers_episodes_evenly():
    sampler = BalancedEpisodeSampler(num_episodes=4, samples_per_rank=8, rank=0, world_size=2, seed=0)
    counts = np.bincount(list(sampler), minlength=4)
    assert counts.sum() == 8
    other = BalancedEpisodeSampler(num_episodes=4, samples_per_rank=8, rank=1, world_size=2, seed=0)
    total = counts + np.bincount(list(other), minlength=4)
    assert (total == 4).all(), "every episode is drawn equally often across ranks"
