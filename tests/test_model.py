import torch

from boxwam.model.boxwam import BoxWAM, load_predictor_state
from boxwam.model.flow import euler_schedule, integrate, interpolate, sample_timesteps


def inputs(model, batch=2):
    torch.manual_seed(0)
    patches = torch.randn(batch, model.num_patches, 32)
    state = torch.randn(batch, model.state_dim)
    actions = torch.randn(batch, model.horizon, model.action_dim)
    boxes = torch.randn(batch, model.horizon, model.box_dim) if model.has_boxes else None
    return patches, state, actions, boxes, torch.rand(batch)


def test_forward_shapes(tiny_model):
    patches, state, actions, boxes, t = inputs(tiny_model)
    velocity = tiny_model(patches, state, actions, boxes, t)
    assert velocity.action.shape == (2, 3, 4)
    assert velocity.box.shape == (2, 3, 16)


def test_action_only_model(tiny_kwargs):
    model = BoxWAM(**tiny_kwargs, box_dim=0).eval()
    patches, state, actions, _, t = inputs(model)
    velocity = model(patches, state, actions, None, t)
    assert velocity.action.shape == (2, 3, 4) and velocity.box is None
    cache = model.encode_observation(patches, state)
    cached = model.predict_velocity(cache, state, actions, None, t)
    torch.testing.assert_close(cached.action, velocity.action, atol=1e-5, rtol=1e-4)


def test_cached_path_matches_full_forward(tiny_kwargs):
    for flags in [(True, False), (False, True), (True, True), (False, False)]:
        model = BoxWAM(**tiny_kwargs, box_dim=16, action_attend_box=flags[0], box_attend_action=flags[1]).eval()
        patches, state, actions, boxes, t = inputs(model)
        with torch.no_grad():
            full = model(patches, state, actions, boxes, t)
            cached = model.predict_velocity(model.encode_observation(patches, state), state, actions, boxes, t)
        torch.testing.assert_close(cached.action, full.action, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(cached.box, full.box, atol=1e-5, rtol=1e-4)


def test_attention_mask_follows_flags(tiny_kwargs):
    model = BoxWAM(**tiny_kwargs, box_dim=16, action_attend_box=True, box_attend_action=False)
    mask = model._attention_mask(True, torch.device("cpu"))[0, 0]
    n, h = model.num_patches, model.horizon
    obs, act, box = slice(0, n), slice(n, n + h), slice(n + h, n + 2 * h)
    assert mask[obs, obs].all() and not mask[obs, act].any() and not mask[obs, box].any()
    assert mask[act, obs].all() and mask[act, act].all() and mask[act, box].all()
    assert mask[box, obs].all() and mask[box, box].all() and not mask[box, act].any()


def test_box_tokens_do_not_influence_actions_when_separate(tiny_kwargs):
    model = BoxWAM(**tiny_kwargs, box_dim=16, action_attend_box=False, box_attend_action=False).eval()
    patches, state, actions, boxes, t = inputs(model)
    with torch.no_grad():
        a = model(patches, state, actions, boxes, t).action
        b = model(patches, state, actions, boxes + 1.0, t).action
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_wrapper_prefixes_are_stripped(tiny_model):
    state = {f"module._orig_mod.{key}": value for key, value in tiny_model.state_dict().items()}
    load_predictor_state(tiny_model, state)


def test_unknown_keys_are_rejected(tiny_model):
    state = dict(tiny_model.state_dict())
    state["something_else"] = torch.zeros(1)
    try:
        load_predictor_state(tiny_model, state)
    except RuntimeError:
        return
    raise AssertionError("unexpected key was accepted")


def test_flow_helpers():
    schedule = euler_schedule(5)
    assert schedule[0][0] == 0.0
    assert abs(sum(dt for _, dt in schedule) - 1.0) < 1e-12
    assert all(schedule[i][1] > schedule[i + 1][1] for i in range(4)), "steps shrink towards t = 1"
    t = sample_timesteps(10_000, device=torch.device("cpu"))
    assert t.min() >= 0 and t.max() <= 0.999 and t.mean() < 0.5, "Beta(1.5, 1) favours small t"
    target, noise = torch.ones(2, 3, 4), torch.zeros(2, 3, 4)
    x_t, velocity = interpolate(target, noise, torch.tensor([0.25, 1.0]))
    assert torch.allclose(x_t[0], torch.full((3, 4), 0.25)) and torch.allclose(x_t[1], target[1])
    assert torch.allclose(velocity, target - noise)


def test_integrate_returns_finite_chunks(tiny_model):
    patches, state, actions, boxes, _ = inputs(tiny_model)
    cache = tiny_model.encode_observation(patches, state)
    actions, boxes = integrate(tiny_model, cache, state, actions, boxes, num_steps=4)
    assert torch.isfinite(actions).all() and torch.isfinite(boxes).all()
