import glob
import os
import os.path as osp
import subprocess
import sys
import tempfile

import pytest


run_examples = (
    os.environ.get("RUN_EXAMPLE_TESTS", "false").lower() == "true"
    or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"
)

pytestmark = pytest.mark.skipif(
    not run_examples,
    reason="Skipping example tests"
           "unless RUN_EXAMPLE_TESTS is set or running in GitHub Actions"
)

examples_dir = osp.join(osp.dirname(__file__), "..", "..", "examples")
example_scripts = sorted(glob.glob(osp.join(examples_dir, "*.py")))


@pytest.mark.parametrize(
    "script", example_scripts,
    ids=[osp.basename(s) for s in example_scripts],
)
def test_example(script):
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = os.environ.copy()
            env["TMPDIR"] = tmp_dir

            cmd = [sys.executable, script, "--no-interactive"]
            print("Executing: {} (attempt {}/{})".format(
                " ".join(cmd), attempt, max_attempts))
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            if result.returncode == 0:
                print("Success on attempt {}".format(attempt))
                return

            if attempt < max_attempts:
                print("Failed on attempt {}, retrying...".format(attempt))

    pytest.fail(
        "Script {} failed with exit code {}\nstdout:\n{}\nstderr:\n{}".format(
            script, result.returncode,
            result.stdout.decode(), result.stderr.decode())
    )
