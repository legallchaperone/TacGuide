"""Full-size checks of plan items 2.5 and 2.6 on one GPU, before A and B are trained.

    CUDA_VISIBLE_DEVICES=7 $TG/venv/bin/python training/tacguide/checks/check_real_weights.py --task pd

The pi0.5 base weights stand in for A (with AnyTouch2) and B, and one real batch is
read from the task's converted dataset. It verifies, at full model size and in the
serving precision:

  2.6  guided sampling with w = 1 equals the strong model alone, element for element;
       with w = 0 it equals the weak model alone
  2.5  the training target equals v_A at w = 1 and v_B at w = 0; a student with B's
       weights has exactly zero loss at w = 0; teacher hashes are unchanged

Pass `--strong-dir` / `--weak-dir` to run the same checks on trained checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

_TRAINING_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TRAINING_DIR))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="gc")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--strong-dir")
    parser.add_argument("--weak-dir")
    parser.add_argument("--out")
    args = parser.parse_args()

    from tacguide import configs
    from tacguide import train_distill
    from tacguide import velocity

    import jax
    import torch
    import torch.nn.functional as F  # noqa: N812

    import openpi.training.data_loader as _data

    from tacguide.distill import DistillStudent
    from tacguide.guided import guided_sample_actions

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    strong_config = configs.base_config(f"{args.task}_A")
    weak_config = configs.base_config(f"{args.task}_B")
    student_config = dataclasses.replace(
        configs.student_config(f"{args.task}_D_w2"), batch_size=args.batch_size, num_workers=0
    )
    standin = args.strong_dir is None or args.weak_dir is None
    base = pathlib.Path(weak_config.pytorch_weight_path)
    strong = train_distill.load_teacher(strong_config, args.strong_dir or base, device, base_standin=standin)
    weak = train_distill.load_teacher(weak_config, args.weak_dir or base, device, base_standin=standin)

    loader = _data.create_data_loader(student_config, framework="pytorch", shuffle=False)
    observation, actions = next(iter(loader))
    observation = jax.tree.map(lambda x: x.to(device), observation)
    actions = actions.to(torch.float32).to(device)
    results = {"task": args.task, "base_standin": standin, "batch_image_keys": list(observation.images)}

    # ---- 2.6: guided sampling against each model alone
    weak_observation = velocity.vision_observation(observation)
    torch.manual_seed(0)
    noise = strong.sample_noise(actions.shape, device)
    only_strong = strong.sample_actions(device, observation, noise=noise, num_steps=10)
    only_weak = weak.sample_actions(device, weak_observation, noise=noise, num_steps=10)

    def guided(w):
        return guided_sample_actions(strong, weak, device, observation, weak_observation, w=w, noise=noise, num_steps=10)

    results["guided_w1_equals_strong"] = bool(torch.equal(guided(1.0), only_strong))
    results["guided_w0_equals_weak"] = bool(torch.equal(guided(0.0), only_weak))
    results["strong_vs_weak_max_abs_diff"] = float((only_strong - only_weak).abs().max())
    results["guided_w2_differs_from_strong"] = not bool(torch.equal(guided(2.0), only_strong))

    # ---- 2.5: training target, with a student that carries B's weights
    student = DistillStudent(dataclasses.replace(weak_config.model, dtype=student_config.pytorch_training_precision))
    student.load_state_dict(weak.state_dict())
    student.to(device).train()
    student.gradient_checkpointing_enable()
    keys = tuple(strong_config.model.image_keys)
    student.attach_teachers(strong=strong, weak=weak, w=2.0, batch_image_keys=keys)
    hashes = student.teacher_hashes()
    time = student.sample_time(actions.shape[0], device)
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions

    torch.manual_seed(1)
    processed = velocity.preprocess_once(observation, keys, train=True)
    with torch.no_grad():
        v_a = velocity.predict_velocity(strong, velocity.full_view(processed, keys), x_t, time)
        v_b = velocity.predict_velocity(weak, velocity.vision_view(processed), x_t, time)
        v_s = velocity.predict_velocity(student, velocity.vision_view(processed), x_t, time)

    def loss_at(w):
        student.attach_teachers(strong=strong, weak=weak, w=w, batch_image_keys=keys)
        torch.manual_seed(1)
        with torch.no_grad():
            return student(observation, actions, noise=noise, time=time)

    results["target_w1_equals_vA"] = bool(torch.equal(loss_at(1.0), F.mse_loss(v_a, v_s, reduction="none")))
    results["target_w0_equals_vB"] = bool(torch.equal(loss_at(0.0), F.mse_loss(v_b, v_s, reduction="none")))
    results["student_equal_to_B_loss_at_w0"] = float(loss_at(0.0).max())
    results["vA_vs_vB_max_abs_diff"] = float((v_a - v_b).abs().max())
    results["student_has_teacher_weights_in_state_dict"] = len(student.state_dict()) != len(weak.state_dict())
    student.attach_teachers(strong=strong, weak=weak, w=2.0, batch_image_keys=keys)
    results["teacher_hashes_unchanged"] = student.teacher_hashes() == hashes
    if torch.cuda.is_available():
        results["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    expected_true = [
        "guided_w1_equals_strong", "guided_w0_equals_weak", "guided_w2_differs_from_strong",
        "target_w1_equals_vA", "target_w0_equals_vB", "teacher_hashes_unchanged",
    ]
    results["passed"] = (
        all(results[key] for key in expected_true)
        and results["student_equal_to_B_loss_at_w0"] == 0.0
        and not results["student_has_teacher_weights_in_state_dict"]
        and results["vA_vs_vB_max_abs_diff"] > 0
    )
    print(json.dumps(results, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    raise SystemExit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
