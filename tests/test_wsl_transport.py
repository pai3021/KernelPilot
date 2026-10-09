from __future__ import annotations

import os
import unittest

from benchmark_backend import RemoteSSHConfig
from spawn import remote_transport_toml_lines


class WslTransportConfigTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "WSL-native transport is POSIX-only")
    def test_spawn_selects_native_openssh_for_posix_control_plane(self):
        lines = remote_transport_toml_lines()
        self.assertIn('ssh_binary = "/usr/bin/ssh"', lines)
        self.assertIn('scp_binary = "/usr/bin/scp"', lines)
        self.assertFalse(any(".exe" in line for line in lines))

        values = {
            "remote_host": "example-gpu", "remote_repo": "/srv/KernelPilot",
            "remote_python": "/srv/python", "remote_runs_dir": "/srv/runs",
            "remote_kernelbench": "/srv/KernelBench",
            "ssh_binary": "/usr/bin/ssh", "scp_binary": "/usr/bin/scp",
        }
        config = RemoteSSHConfig.from_mapping(values)
        self.assertEqual((config.ssh_binary, config.scp_binary), ("/usr/bin/ssh", "/usr/bin/scp"))
