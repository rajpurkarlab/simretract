"""CPU regressions against installed Ultralytics state initialization."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import inspect
from unittest.mock import patch

import numpy as np
import torch
try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError:
    SAM3VideoSemanticPredictor = None

from simretract.real import mask_server as server



class FakeTrackerModel:
    def __init__(self):
        self.model = object()
        self.inference_state = {}
        self._bb_feat_sizes = [(16, 16)]
        self.propagated = []
        self.reset_image()

    def reset_image(self):
        self.im = self.features = None

    def propagate_in_video(self, state, frame_idx):
        self.propagated.append(frame_idx)
        state['old_warm_memory'] = object()


class FakePredictor:
    @staticmethod
    def init_state(p):
        if SAM3VideoSemanticPredictor is not None:
            return SAM3VideoSemanticPredictor.init_state(p)
        p.inference_state = {
            "text_prompt": None, "tracker_inference_states": [], "tracker_metadata": {},
            "per_frame_geometric_prompt": [None] * p.dataset.frames,
            "num_frames": p.dataset.frames,
        }
    constructed = 0

    def __init__(self, overrides=None):
        type(self).constructed += 1
        self.model = SimpleNamespace(names=["metal rake"], text_embeddings=object())
        self.tracker = FakeTrackerModel()
        self.inference_state = {}
        self.dataset = SimpleNamespace(mode='video', frame=1, frames=2,
                                       cap=SimpleNamespace(release=lambda: None))
        self.device = torch.device('cpu')
        self.setup_model_calls = self.setup_source_calls = 0
        self.outputs = []
        self.prompts_seen = []
        self.synthetic_masks = []
        self.reset_image()

    def reset_image(self):
        self.im = self.features = None

    def setup_model(self, model=None):
        self.setup_model_calls += 1

    def setup_source(self, source):
        assert Path(source).exists()
        self.setup_source_calls += 1

    def preprocess(self, frames):
        return torch.zeros((1, 3, 16, 16))

    def inference(self, im, text=None):
        self.prompts_seen.append(text)
        self.tracker.backbone_out = {'new_frame_features': object()}
        return object()

    def postprocess(self, predictions, im, bgr):
        return [self.outputs.pop(0) if self.outputs else SimpleNamespace(masks=None, boxes=None)]

    def _tracker_add_new_objects(self, **kwargs):
        self.synthetic_masks.append(kwargs['new_obj_masks'].clone())
        return [{'num_frames': kwargs['num_frames'], 'obj_ids': kwargs['new_obj_ids']}]


def result(*candidates):
    if not candidates:
        return SimpleNamespace(masks=None, boxes=None)
    ids, masks = zip(*candidates)
    class InstanceMasks:
        data = torch.from_numpy(np.stack(masks))

        def __len__(self):
            return len(masks)

    return SimpleNamespace(masks=InstanceMasks(), boxes=SimpleNamespace(id=torch.tensor(ids)))


def rectangle(y0, y1, x0, x1):
    mask = np.zeros((32, 40), np.float32)
    mask[y0:y1, x0:x1] = 1
    return mask


class WarmResetTests(unittest.TestCase):
    def setUp(self):
        self.tracker = server.Tracker("weights.pt", "metal rake", .2)
        self.rgb = np.zeros((32, 40, 3), np.uint8)

    def load(self):
        self.tracker.pred = FakePredictor()
        self.tracker.reset()
        return self.tracker.pred

    def test_reset_retains_models_and_clears_every_episode_cache(self):
        p = self.load()
        model, interactive, embedding = p.model, p.tracker.model, p.model.text_embeddings
        for _ in range(200):
            p.inference_state.update(text_ids=object(), text_prompt='old', im=object(),
                                     tracker_inference_states=[{'old_memory': object()}],
                                     tracker_metadata={'obj_ids': [91]})
            for pred in (p, p.tracker):
                pred.im = pred.features = pred.backbone_out = object()
                pred.prompts = {'old_box': object()}
                pred.batch = pred.results = object()
            self.tracker.lock_id, self.tracker.lock_mask, self.tracker.lost = 91, self.rgb, 25
            self.tracker.reset()
            self.assertIs(self.tracker.pred, p)
            self.assertIs(p.model, model)
            self.assertIs(p.tracker.model, interactive)
            self.assertIs(p.model.text_embeddings, embedding)
            self.assertNotIn('text_ids', p.inference_state)
            self.assertIsNone(p.inference_state['text_prompt'])
            self.assertEqual(p.inference_state['tracker_inference_states'], [])
            self.assertEqual(p.inference_state['tracker_metadata'], {})
            self.assertEqual(len(p.inference_state['per_frame_geometric_prompt']), 256)
            self.assertEqual(p.dataset.frame, 1)
            for pred in (p, p.tracker):
                self.assertIsNone(pred.im)
                self.assertIsNone(pred.features)
                self.assertIsNone(pred.backbone_out)
                self.assertEqual(pred.prompts, {})
                self.assertIsNone(pred.batch)
                self.assertIsNone(pred.results)
            self.assertIsNone(self.tracker.lock_id)
            self.assertIsNone(self.tracker.lock_mask)
            self.assertEqual(self.tracker.lost, 0)
            self.assertTrue(self.tracker.first)
        self.assertEqual(p.setup_model_calls, 0)
        self.assertEqual(p.setup_source_calls, 0)

    def test_growing_capacity_preserves_prompts_and_updates_nested_horizons(self):
        p = self.load()
        prompt = object()
        p.inference_state['per_frame_geometric_prompt'][0] = prompt
        states = [{'num_frames': 256}, {'num_frames': 256}]
        p.inference_state['tracker_inference_states'] = states
        p.dataset.frame = 257
        self.tracker._ensure_frame_capacity()
        self.assertEqual(p.dataset.frames, 512)
        self.assertEqual([s['num_frames'] for s in states], [512, 512])
        self.assertIs(p.inference_state['per_frame_geometric_prompt'][0], prompt)
        p.dataset.frame = 4099
        self.tracker._ensure_frame_capacity()
        self.assertEqual(p.inference_state['num_frames'], 4099)
        self.assertEqual(len(p.inference_state['per_frame_geometric_prompt']), 4099)
        self.assertEqual([s['num_frames'] for s in states], [4099, 4099])
        self.tracker.reset()
        self.assertEqual(p.dataset.frames, 256)

    def test_first_real_frame_after_reset_never_reloads_model(self):
        p = self.load()
        for _ in range(3):
            self.tracker.mask(self.rgb)
            self.tracker.reset()
        self.assertEqual(p.setup_model_calls, 0)
        self.assertEqual(p.setup_source_calls, 0)
        self.assertEqual(p.prompts_seen, [["metal rake"]] * 3)

    def test_warmup_runs_temporal_path_then_discards_synthetic_state(self):
        self.load()
        self.tracker.warmup()
        p = self.tracker.pred
        self.assertEqual(p.tracker.propagated, [1])
        self.assertEqual(len(p.synthetic_masks), 1)
        self.assertGreater(int(p.synthetic_masks[0].sum()), 0)
        self.assertEqual(p.inference_state['tracker_inference_states'], [])
        self.assertEqual(p.inference_state['tracker_metadata'], {})
        self.assertIsNone(p.tracker.backbone_out)
        self.assertEqual(p.dataset.frame, 1)
        self.assertIsNone(self.tracker.lock_id)

    def test_locked_id_and_gripper_filter_are_preserved(self):
        p = self.load()
        gripper = rectangle(0, 20, 0, 20)
        tool = rectangle(22, 30, 20, 35)
        p.outputs = [result((1, gripper), (7, tool)), result((1, gripper)),
                     result((7, tool), (1, gripper))]
        out = self.tracker.mask(self.rgb, gripper.astype(np.uint8))
        self.assertEqual(self.tracker.lock_id, 7)
        self.assertTrue(np.array_equal(out > 0, tool > 0))
        self.assertEqual(np.count_nonzero(self.tracker.mask(self.rgb)), 0)
        self.assertEqual(self.tracker.lock_id, 7)
        self.assertTrue(np.array_equal(self.tracker.mask(self.rgb) > 0, tool > 0))
        self.assertEqual(p.prompts_seen, [["metal rake"], None, None])

    def test_reset_allows_new_episode_target_and_keeps_disconnected_pieces(self):
        p = self.load()
        both = rectangle(0, 5, 0, 8) + rectangle(20, 25, 20, 28)
        p.outputs = [result((7, both)), result((18, both))]
        self.assertTrue(np.array_equal(self.tracker.mask(self.rgb) > 0, both > 0))
        self.tracker.reset()
        self.assertTrue(np.array_equal(self.tracker.mask(self.rgb) > 0, both > 0))
        self.assertEqual(self.tracker.lock_id, 18)


class LifetimePredictor(FakePredictor):
    if SAM3VideoSemanticPredictor is not None:
        _initialize_metadata = SAM3VideoSemanticPredictor._initialize_metadata
        _process_hotstart = SAM3VideoSemanticPredictor._process_hotstart

    def __init__(self, overrides=None):
        super().__init__(overrides)
        defaults = inspect.signature(SAM3VideoSemanticPredictor.__init__).parameters
        for name in ('hotstart_delay', 'hotstart_unmatch_thresh', 'hotstart_dup_thresh',
                     'init_trk_keep_alive', 'max_trk_keep_alive', 'min_trk_keep_alive',
                     'decrease_trk_keep_alive_for_empty_masklets', 'masklet_confirmation_enable'):
            setattr(self, name, defaults[name].default)
        self.frames = []
        self.removals = []

    def queue_frame(self, *masks, new=(), unmatched=(), empty=(), matched=()):
        self.frames.append((masks, new, unmatched, empty, matched))

    def inference(self, im, text=None):
        prediction = super().inference(im, text=text)
        masks, new, unmatched, empty, matched = self.frames.pop(0)
        state = self.inference_state
        if not state['tracker_metadata']:
            state['tracker_metadata'] = self._initialize_metadata()
        tracker_meta = state['tracker_metadata']
        removed, metadata = self._process_hotstart(
            frame_idx=self.dataset.frame - 1, reverse=False,
            det_to_matched_trk_obj_ids={i: np.array([obj_id]) for i, obj_id in enumerate(matched)},
            new_det_obj_ids=np.array(new, dtype=np.int64),
            empty_trk_obj_ids=np.array(empty, dtype=np.int64),
            unmatched_trk_obj_ids=np.array(unmatched, dtype=np.int64),
            metadata=tracker_meta['metadata'])
        active = (set(tracker_meta['obj_ids']) | set(new)) - removed
        tracker_meta['obj_ids'] = np.array(sorted(active), dtype=np.int64)
        tracker_meta['metadata'] = metadata
        self.removals.append(set(removed))
        # Emulate removal from semantic outputs after its actual lifetime rule.
        self.outputs.append(result(*(pair for pair in masks if pair[0] in active)))
        return prediction

    @property
    def active(self):
        return set(self.inference_state['tracker_metadata']['obj_ids'])


@unittest.skipUnless(SAM3VideoSemanticPredictor is not None, "requires the mask environment")
class LockedTrackTests(unittest.TestCase):
    def setUp(self):
        self.tracker = server.Tracker("weights.pt", "metal rake", .2)
        self.rgb = np.zeros((32, 40, 3), np.uint8)
        self.tracker.pred = LifetimePredictor()
        self.tracker.reset()
        self.p = self.tracker.pred
        self.tool = rectangle(3, 20, 3, 18)
        self.other = rectangle(22, 29, 25, 34)

    def acquire(self):
        self.p.queue_frame((7, self.tool), (9, self.other), new=(7, 9))
        self.tracker.mask(self.rgb)
        self.assertEqual(self.tracker.lock_id, 7)

    def test_selected_id_survives_35_detector_misses_while_other_id_is_removed(self):
        self.acquire()
        for index in range(35):
            fresh = rectangle(3, 20, 4 + index % 3, 19 + index % 3)
            self.p.queue_frame((7, fresh), (9, self.other), unmatched=tuple(self.p.active))
            output = self.tracker.mask(self.rgb)
            self.assertIn(7, self.p.active)
            np.testing.assert_array_equal(output > 0, fresh > 0)
        self.assertNotIn(9, self.p.active)
        self.assertEqual(set().union(*self.p.removals), {9})
        self.assertEqual(self.p.removals[10], {9})
        self.assertEqual(self.p.prompts_seen, [["metal rake"]] + [None] * 35)

    def test_empty_outputs_stay_empty_without_freezing_or_selecting_another_id(self):
        self.acquire()
        # Establish unmatched history, so subsequent empty masks exercise the
        # actual deletion condition, not just the empty-only bookkeeping case.
        self.p.queue_frame((9, self.other), unmatched=(7,), matched=(9,))
        self.assertFalse(self.tracker.mask(self.rgb).any())
        for _ in range(25):
            self.p.queue_frame((9, self.other), empty=(7,), matched=(9,))
            self.assertFalse(self.tracker.mask(self.rgb).any())
            self.assertEqual(self.tracker.lock_id, 7)
            self.assertIn(7, self.p.active)
        self.assertEqual(self.tracker.lost, 26)
        returned = rectangle(12, 27, 17, 31)
        self.p.queue_frame((7, returned), (9, self.other), matched=(7, 9))
        np.testing.assert_array_equal(self.tracker.mask(self.rgb) > 0, returned > 0)
        self.assertEqual(self.tracker.lost, 0)
        self.assertEqual(self.p.prompts_seen[1:], [None] * 27)

    def test_reset_removes_previous_id_protection_and_allows_new_episode_selection(self):
        self.acquire()
        self.tracker.reset()
        self.assertIsNone(self.tracker.lock_id)
        self.assertEqual(self.p.inference_state['tracker_metadata'], {})
        # Reusing numeric ID 7 in a new episode must not carry its old privilege.
        self.p.queue_frame((7, self.other), (18, self.tool), new=(7, 18))
        self.tracker.mask(self.rgb)
        self.assertEqual(self.tracker.lock_id, 18)
        for _ in range(15):
            self.p.queue_frame((7, self.other), (18, self.tool), unmatched=tuple(self.p.active))
            self.tracker.mask(self.rgb)
        self.assertEqual(self.p.active, {18})
        self.assertIn({7}, self.p.removals)
        self.assertEqual(self.p.prompts_seen[:2], [["metal rake"], ["metal rake"]])

    def test_unselected_track_has_no_lifetime_protection(self):
        self.p.queue_frame(new=(7,))
        self.assertFalse(self.tracker.mask(self.rgb).any())
        for _ in range(10):
            self.p.queue_frame(unmatched=(7,))
            self.assertFalse(self.tracker.mask(self.rgb).any())
        self.assertIsNone(self.tracker.lock_id)
        self.assertNotIn(7, self.p.active)
        self.assertEqual(self.p.removals[-1], {7})


if __name__ == "__main__":
    unittest.main()
