# Release scope

## Included

- friction-conditioned PPO tasks and rewards;
- Oracle Teacher with privileged effective friction;
- 640-D Student distillation and DAgger;
- shared dual-foot Hall encoder experiments (`1864 → 548 → 29`);
- structured sensor noise, dropout and Sim2Sim randomization;
- Isaac Lab and MuJoCo friction/speed evaluators;
- C++ foot bridge, gamepad control and conservative traction governor;
- selected result reports, figures, demos and representative checkpoints.

## Intentionally excluded

- all `logs/`, TensorBoard event files and raw policy-observation binaries;
- multi-hundred-megabyte `.npz` rollout and DAgger datasets;
- repetitive checkpoints saved every 10–50 iterations;
- rejected model variants and duplicated deployment slots;
- editable paper PPT files, temporary frames and local presentation assets;
- machine-specific BLE/Hall calibration and real-floor classifier;
- SDK, MuJoCo and Isaac Sim installations already available upstream.

These files remain untouched in the original local workspace. Exclusion from
GitHub is implemented through selective copying and `.gitignore`, not deletion.
