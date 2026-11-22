"""CPU check: the student's data pipeline against the vision-only pipeline, on real converted data.

    CUDA_VISIBLE_DEVICES= $TG/venv/bin/python training/tacguide/checks/check_data_view.py --task pd

The distilled student reads batches from the vision + tactile pipeline. This confirms
that the camera images, state, prompt tokens and actions it (and teacher B) get out of
such a batch are identical to what the vision-only pipeline feeds B in its own training,
and that the two models' normalisation statistics agree.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gc")
    args = parser.parse_args()

    from tacguide import configs
    from tacguide import velocity

    import torch

    import openpi.training.data_loader as _data

    strong, weak = configs.base_config(f"{args.task}_A"), configs.base_config(f"{args.task}_B")
    student = dataclasses.replace(configs.student_config(f"{args.task}_D_w2"), batch_size=2, num_workers=0)
    out = {
        "student_checkpoint_dir": str(student.checkpoint_dir),
        "student_steps": student.num_train_steps,
        "student_batch_size_default": configs.student_config(f"{args.task}_D_w2").batch_size,
        "b_steps": weak.num_train_steps,
        "b_batch_size": weak.batch_size,
        "student_model_keys": list(student.model.image_keys),
        "teacher_model_keys": list(strong.model.image_keys),
    }
    stats = [c.data.create(c.assets_dirs, c.model).norm_stats for c in (strong, weak, student)]
    out["norm_stats_A_equals_B"] = velocity.norm_stats_equal(stats[0], stats[1])[0]
    out["norm_stats_student_equals_A"] = velocity.norm_stats_equal(stats[2], stats[0])[0]

    observation, actions = next(iter(_data.create_data_loader(student, framework="pytorch", shuffle=False)))
    out["batch_images"] = {k: list(v.shape) for k, v in observation.images.items()}
    view = velocity.vision_view(velocity.preprocess_once(observation, strong.model.image_keys, train=False))
    out["student_view_keys"] = list(view.image_keys)
    out["student_view_masks"] = {k: v.tolist() for k, v in view.image_masks.items()}

    weak_loader = _data.create_data_loader(
        dataclasses.replace(weak, batch_size=2, num_workers=0), framework="pytorch", shuffle=False
    )
    reference, reference_actions = next(iter(weak_loader))
    derived = velocity.vision_observation(observation)
    out["same_pixels"] = {k: bool(torch.equal(reference.images[k], derived.images[k])) for k in reference.images}
    out["same_masks"] = {k: bool(torch.equal(reference.image_masks[k], derived.image_masks[k])) for k in reference.images}
    out["same_state"] = bool(torch.equal(reference.state, observation.state))
    out["same_prompt_tokens"] = bool(torch.equal(reference.tokenized_prompt, observation.tokenized_prompt))
    out["same_actions"] = bool(torch.equal(reference_actions, actions))
    out["passed"] = bool(
        out["norm_stats_A_equals_B"] and out["norm_stats_student_equals_A"]
        and all(out["same_pixels"].values()) and all(out["same_masks"].values())
        and out["same_state"] and out["same_prompt_tokens"] and out["same_actions"]
        and not any("tactile" in k for k in view.image_keys)
    )
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
