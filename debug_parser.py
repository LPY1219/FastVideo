#!/usr/bin/env python3
import sys
from fastvideo.fastvideo_args import TrainingArgs, FastVideoArgs
from fastvideo.utils import FlexibleArgumentParser

# Build parser exactly like wan_distillation_pipeline.py
parser = FlexibleArgumentParser()
print("Step 1: Created FlexibleArgumentParser")

parser = TrainingArgs.add_cli_args(parser)
print("Step 2: Added TrainingArgs")

# Check if arguments exist after TrainingArgs
actions_after_training = {action.dest: action for action in parser._actions}
if 'real_score_model_path' in actions_after_training:
    print(f"  ✓ real_score_model_path exists: {actions_after_training['real_score_model_path'].option_strings}")
else:
    print("  ✗ real_score_model_path NOT found")

parser = FastVideoArgs.add_cli_args(parser)
print("Step 3: Added FastVideoArgs")

# Check if arguments exist after FastVideoArgs
actions_after_fastvideo = {action.dest: action for action in parser._actions}
if 'real_score_model_path' in actions_after_fastvideo:
    print(f"  ✓ real_score_model_path exists: {actions_after_fastvideo['real_score_model_path'].option_strings}")
else:
    print("  ✗ real_score_model_path NOT found")

print("\nAll registered model-path arguments:")
for dest, action in actions_after_fastvideo.items():
    if 'model' in dest and 'path' in dest:
        print(f"  {dest}: {action.option_strings}")

print("\nAttempting to parse test arguments...")
test_args = [
    '--model-path', '/tmp/test',
    '--real-score-model-path', '/tmp/test2',
    '--data-path', '/tmp',
    '--dataloader-num-workers', '0',
    '--num-height', '448',
    '--num-width', '832',
    '--num-frames', '77',
    '--train-batch-size', '1',
    '--num-latent-t', '20',
    '--pretrained-model-name-or-path', '/tmp/test',
    '--output-dir', '/tmp/out',
    '--learning-rate', '1e-5'
]

try:
    args = parser.parse_args(test_args)
    print("✓ Success! Arguments parsed correctly")
    print(f"  real_score_model_path = {args.real_score_model_path}")
except SystemExit as e:
    print(f"✗ Failed with SystemExit: {e}")
except Exception as e:
    print(f"✗ Failed with exception: {e}")
