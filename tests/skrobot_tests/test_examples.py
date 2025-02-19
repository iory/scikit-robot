import os
import os.path as osp
import subprocess
import sys
import tempfile
import unittest

import pytest


run_examples = (
    os.environ.get("RUN_EXAMPLE_TESTS", "false").lower() == "true"
    or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"
)

pytestmark = pytest.mark.skipif(
    not run_examples,
    reason="Skipping example tests"
           + "unless RUN_EXAMPLE_TESTS is set or running in GitHub Actions"
)


class TestExampleScripts(unittest.TestCase):

    @pytest.mark.skipif(sys.version_info[0] == 2, reason="Skip in Python 2")
    def test_all_examples(self):
        examples_dir = osp.join(osp.dirname(__file__), "..", "..", "examples")
        self.assertTrue(osp.exists(examples_dir),
                        "Examples directory not found: {}"
                        .format(examples_dir))

        example_scripts = [
            osp.join(examples_dir, f)
            for f in os.listdir(examples_dir)
            if f.endswith(".py")
        ]
        self.assertTrue(len(example_scripts) > 0,
                        "No example scripts found in examples/ directory")

        failures = []

        for script in example_scripts:
            with tempfile.TemporaryDirectory() as tmp_dir:
                env = os.environ.copy()
                env["TMPDIR"] = tmp_dir

                cmd = "{} {} --no-interactive".format(sys.executable, script)
                print("Executing: {}".format(cmd))
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )

                if result.returncode != 0:
                    failures.append(
                        (
                            script,
                            result.returncode,
                            result.stdout.decode(),
                            result.stderr.decode(),
                        )
                    )

        if failures:
            messages = []
            for script, code, stdout, stderr in failures:
                messages.append(
                    "Script {} failed with exit ".format(script)
                    + "code {}\nstdout:\n{}\nstderr:\n{}".format(
                        code, stdout, stderr)
                )
            self.fail("\n\n".join(messages))
