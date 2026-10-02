"""Reject platform topology mistakes before starting distributed workers."""

import unittest

from scripts.launch_b200 import resolve_topology


class PlatformTopologyTests(unittest.TestCase):
    def test_five_nodes_with_automatic_gpu_count(self):
        for mode in ("auto", "gpu"):
            for rank in (0, 4):
                with self.subTest(mode=mode, rank=rank):
                    env = dict(NNODES="5", NPROC_PER_NODE=mode, WORLD_SIZE="40", NODE_RANK=str(rank))
                    self.assertEqual(resolve_topology("multi", env, visible_gpu_count=8), (5, 8, rank))

    def test_auto_respects_explicit_gpu_count(self):
        env = dict(NNODES="5", NPROC_PER_NODE="auto", GPUS_PER_NODE="4", WORLD_SIZE="20", NODE_RANK="0")
        self.assertEqual(resolve_topology("multi", env, visible_gpu_count=8), (5, 4, 0))

    def test_auto_uses_visible_devices_and_rejects_cpu_fallback(self):
        env = dict(NNODES="5", NPROC_PER_NODE="auto", NODE_RANK="0")
        self.assertEqual(resolve_topology("multi", env, visible_gpu_count=4), (5, 4, 0))
        for count in (0, 1):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "at least 2 GPUs"):
                resolve_topology("multi", env, visible_gpu_count=count)

    def test_seven_node_platform(self):
        for rank in range(7):
            with self.subTest(rank=rank):
                env = dict(NNODES="7", NPROC_PER_NODE="8", WORLD_SIZE="56", NODE_RANK=str(rank))
                self.assertEqual(resolve_topology("multi", env), (7, 8, rank))

    def test_legacy_launch_and_rank_precedence(self):
        self.assertEqual(resolve_topology("multi", dict(NNODES="7", GPUS_PER_NODE="8", RANK="6")), (7, 8, 6))
        self.assertEqual(resolve_topology("multi", dict(NNODES="7", NODE_RANK="6", RANK="48")), (7, 8, 6))

    def test_conflicting_or_invalid_platform_values(self):
        base = dict(NNODES="7", NPROC_PER_NODE="8", WORLD_SIZE="56", NODE_RANK="0")
        cases = [
            (dict(WORLD_SIZE="7"), "WORLD_SIZE"),
            (dict(WORLD_SIZE="1"), "WORLD_SIZE"),
            (dict(GPUS_PER_NODE="4"), "conflicts"),
            (dict(NODE_RANK="7"), "NODE_RANK"),
            (dict(NODE_RANK="-1"), "NODE_RANK"),
            (dict(NNODES="auto"), "NNODES"),
            (dict(NPROC_PER_NODE="invalid"), "NPROC_PER_NODE"),
            (dict(NPROC_PER_NODE="1", WORLD_SIZE="7"), "at least 2 GPUs"),
            (dict(LOCAL_RANK="0"), "once per NODE"),
            (dict(OMPI_COMM_WORLD_RANK="0"), "mpirun"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_topology("multi", {**base, **overrides})

    def test_missing_rank_and_unused_mpi_metadata(self):
        with self.assertRaisesRegex(ValueError, "requires NODE_RANK"):
            resolve_topology("multi", dict(NNODES="7"))
        # MPI-related variables may exist in the container image; they neither
        # select MPI launch mode nor override the PyTorch topology.
        env = dict(NNODES="7", NODE_RANK="6", NP="4", OMPI_MCA_orte_defaut_hostfile="/etc/mpi/hostfile")
        self.assertEqual(resolve_topology("multi", env), (7, 8, 6))

    def test_single_and_debug_ignore_platform_topology(self):
        env = dict(NNODES="7", NODE_RANK="6", NPROC_PER_NODE="8", WORLD_SIZE="56", GPUS_PER_NODE="4")
        for mode in ("single", "debug"):
            self.assertEqual(resolve_topology(mode, env), (1, 4, 0))


if __name__ == "__main__":
    unittest.main()
