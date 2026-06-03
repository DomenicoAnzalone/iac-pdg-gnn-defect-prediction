from __future__ import annotations

import csv
import tempfile
import unittest
from queue import Queue
from pathlib import Path
from unittest.mock import Mock, patch

import networkx as nx

from scripts.pdg_file_level_extractor import (
    DatasetRow,
    acquire_run_lock,
    checkout_commit,
    drain_result_queue,
    graph_status_from_dot,
    group_rows_by_repository,
    load_dataset_rows,
    main,
    normalize_filepath,
    normalize_repository_identifier,
    output_path_for_file,
    prepare_scansible_target,
    process_repository_group,
    read_scansible_dot_fallback,
    refuse_concurrent_run,
    remove_run_lock,
    repository_clone_url,
    should_wrap_task_file,
    TERMINAL_STATUSES,
    success_filename_for_input,
    unsupported_file_reason,
    verify_scansible_available,
)


class PdgFileLevelExtractorTests(unittest.TestCase):
    def test_normalize_repository_identifier(self):
        self.assertEqual(
            normalize_repository_identifier("https://github.com/acme/example.git"),
            "acme/example",
        )
        self.assertEqual(normalize_repository_identifier("acme/example"), "acme/example")
        self.assertEqual(
            normalize_repository_identifier("git@github.com:acme/example.git"),
            "acme/example",
        )

    def test_repository_clone_url_uses_repo_url_or_github_identifier(self):
        self.assertEqual(
            repository_clone_url("acme/example"),
            "https://github.com/acme/example.git",
        )
        self.assertEqual(
            repository_clone_url("acme/example", "https://example.test/repo.git"),
            "https://example.test/repo.git",
        )

    def test_success_filename_uses_input_csv_name(self):
        self.assertEqual(
            success_filename_for_input(Path("ansible-2.csv")),
            "ansible-2_rows_successfull_extracted.csv",
        )

    def test_checkout_commit_works_after_no_checkout_clone(self):
        repository_path = Path("/tmp/repository")
        completed = Mock(returncode=0, stdout="", stderr="")

        with patch(
            "scripts.pdg_file_level_extractor.subprocess.run",
            return_value=completed,
        ) as run:
            checkout_commit(repository_path, "abc123")

        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ["git", "-C", str(repository_path), "checkout", "--force", "abc123"],
                ["git", "-C", str(repository_path), "clean", "-fdx"],
            ],
        )

    def test_run_lock_prevents_concurrent_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_file = Path(temp_dir) / "run" / ".run.lock"
            acquire_run_lock(lock_file)
            with self.assertRaises(SystemExit):
                refuse_concurrent_run(lock_file)
            remove_run_lock(lock_file)
            self.assertFalse(lock_file.exists())

    def test_scansible_preflight_fails_fast(self):
        completed = Mock(returncode=1, stdout="", stderr="broken scansible")

        with (
            patch(
                "scripts.pdg_file_level_extractor.subprocess.run",
                return_value=completed,
            ),
            self.assertRaises(SystemExit) as context,
        ):
            verify_scansible_available("scansible")

        self.assertIn("Scansible preflight failed", str(context.exception))

    def test_fallback_dot_parser_reads_scansible_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dot_path = Path(temp_dir) / "pdg.dot"
            dot_path.write_text(
                "digraph {\n"
                "\tnode [fontname=Courier]\n"
                "\t0 [label=<<B>ansible.builtin.assert</B>> fillcolor=lightgrey]\n"
                "\t1 [label=\"str:hello world\" shape=box]\n"
                "\t1 -> 0 [label=\"args.msg\" penwidth=2.5]\n"
                "}\n",
                encoding="utf-8",
            )

            graph = read_scansible_dot_fallback(dot_path)

        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(graph.nodes["0"]["label"], "<<B>ansible.builtin.assert</B>>")
        self.assertEqual(graph.nodes["1"]["label"], "str:hello world")
        edge_attrs = next(iter(graph.edges(data=True)))[2]
        self.assertEqual(edge_attrs["label"], "args.msg")
        self.assertEqual(edge_attrs["penwidth"], "2.5")

    def test_graph_parse_failure_is_retryable(self):
        self.assertNotIn("GRAPH_PARSE_FAILURE", TERMINAL_STATUSES)

    def test_existing_dot_can_be_converted_to_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dot_path = root / "pdg.dot"
            graphml_path = root / "pdg.graphml"
            dot_path.write_text(
                "digraph {\n"
                "\t0 [label=\"source\"]\n"
                "\t1 [label=\"target\"]\n"
                "\t0 -> 1 [label=DEF]\n"
                "}\n",
                encoding="utf-8",
            )

            result = graph_status_from_dot(
                base={
                    "row_index": "1",
                    "repository": "acme/example",
                    "commit": "abc",
                    "filepath": "tasks/main.yml",
                    "failure_prone": "0",
                    "status": "",
                    "nodes": "0",
                    "edges": "0",
                    "graphml_path": "",
                    "error": "",
                },
                dot_path=dot_path,
                graphml_path=graphml_path,
                keep_dot=True,
                min_pdg_nodes=2,
                min_pdg_edges=1,
            )

            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["nodes"], "2")
            self.assertEqual(result["edges"], "1")
            self.assertTrue(graphml_path.exists())

    def test_unresolved_include_placeholder_is_low_quality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dot_path = root / "pdg.dot"
            graphml_path = root / "pdg.graphml"
            dot_path.write_text(
                "digraph {\n"
                "\t0 [label=\"str:tasks/main.yml\"]\n"
                "\t1 [label=<<B>import_tasks</B>>]\n"
                "\t0 -> 1 [label=_raw_params]\n"
                "}\n",
                encoding="utf-8",
            )

            result = graph_status_from_dot(
                base={
                    "row_index": "1",
                    "repository": "acme/example",
                    "commit": "abc",
                    "filepath": "tasks/main.yml",
                    "failure_prone": "0",
                    "status": "",
                    "nodes": "0",
                    "edges": "0",
                    "graphml_path": "",
                    "error": "",
                },
                dot_path=dot_path,
                graphml_path=graphml_path,
                keep_dot=True,
                min_pdg_nodes=2,
                min_pdg_edges=1,
            )

            self.assertEqual(result["status"], "LOW_QUALITY_GRAPH")
            self.assertIn("placeholder", result["error"])
            self.assertFalse(graphml_path.exists())

    def test_load_dataset_accepts_repo_url_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "ansible-2.csv"
            csv_path.write_text(
                "repo_url,commit,filepath,failure_prone\n"
                "https://github.com/acme/example,abc,tasks/main.yml,1\n",
                encoding="utf-8",
            )
            rows = load_dataset_rows(csv_path, max_rows=None)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].repository, "acme/example")
        self.assertEqual(rows[0].clone_url, "https://github.com/acme/example")

    def test_main_creates_run_outputs_and_removes_temporary_clones(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_file = root / "ansible-2.csv"
            output_dir = root / "output"
            input_file.write_text(
                "repo_url,commit,filepath,failure_prone\n"
                "https://github.com/acme/example,abc,tasks/main.yml,1\n",
                encoding="utf-8",
            )

            argv = [
                "pdg_file_level_extractor.py",
                "--input",
                str(input_file),
                "--output-dir",
                str(output_dir),
                "--run-name",
                "second_pdg_extraction",
            ]
            with (
                patch("sys.argv", argv),
                patch("scripts.pdg_file_level_extractor.verify_scansible_available"),
                patch(
                    "scripts.pdg_file_level_extractor.clone_repository",
                    side_effect=RuntimeError("clone failed"),
                ),
            ):
                main()

            run_dir = output_dir / "second_pdg_extraction"
            self.assertTrue((run_dir / "extraction_report.txt").exists())
            self.assertTrue((run_dir / "extraction_status.csv").exists())
            self.assertTrue((run_dir / "pdg_file_level").is_dir())
            self.assertTrue(
                (run_dir / "ansible-2_rows_successfull_extracted.csv").exists()
            )
            self.assertFalse((run_dir / ".repositories").exists())
            self.assertFalse((run_dir / ".run.lock").exists())

            with (run_dir / "extraction_status.csv").open(
                newline="", encoding="utf-8"
            ) as csvfile:
                statuses = list(csv.DictReader(csvfile))
            self.assertEqual(statuses[0]["status"], "CLONE_FAILURE")

    def test_normalize_filepath_rejects_traversal(self):
        self.assertEqual(
            normalize_filepath("roles/web/tasks/main.yml"),
            Path("roles/web/tasks/main.yml"),
        )
        self.assertIsNone(normalize_filepath("../secret.yml"))
        self.assertIsNone(normalize_filepath("/absolute/file.yml"))

    def test_file_type_routing_for_pdg_extraction(self):
        self.assertTrue(should_wrap_task_file(Path("roles/web/tasks/main.yml")))
        self.assertFalse(should_wrap_task_file(Path("playbooks/site.yml")))
        self.assertEqual(
            unsupported_file_reason(Path("roles/web/vars/main.yml")),
            "vars file",
        )
        self.assertEqual(
            unsupported_file_reason(Path("roles/web/defaults/main.yml")),
            "defaults file",
        )
        self.assertEqual(
            unsupported_file_reason(Path("roles/web/handlers/main.yml")),
            "handlers file",
        )

    def test_task_wrapper_imports_target_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            wrapper = prepare_scansible_target(
                repository_path=repo,
                filepath=Path("roles/web/tasks/main.yml"),
                row_index=42,
                use_task_wrapper=True,
            )

            content = wrapper.read_text(encoding="utf-8")

        self.assertIn("- hosts: all", content)
        self.assertIn('- import_tasks: "main.yml"', content)

    def test_grouping_uses_normalized_repository_identifier(self):
        rows = [
            DatasetRow(
                1,
                "acme/shared",
                "c1",
                "tasks/main.yml",
                "0",
                {},
                "https://github.com/acme/shared.git",
            ),
            DatasetRow(
                2,
                "acme/shared",
                "c2",
                "tasks/main.yml",
                "1",
                {},
                "https://github.com/acme/shared.git",
            ),
        ]
        groups = group_rows_by_repository(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual([row.row_index for row in groups[0][1]], [1, 2])

    def test_output_path_preserves_owner_repository_commit_and_file(self):
        path = output_path_for_file(
            Path("/output/pdg_file_level"),
            "acme/example",
            "abc123",
            Path("roles/web/tasks/main.yml"),
        )
        self.assertEqual(
            path,
            Path(
                "/output/pdg_file_level/acme/example/abc123/"
                "roles/web/tasks/main.yml/PDG_FILE_LEVEL"
            ),
        )

    def test_repository_group_processes_rows_sequentially_and_emits_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clone_root = root / ".repositories"

            rows = [
                DatasetRow(
                    1,
                    "acme/example",
                    "c1",
                    "tasks/first.yml",
                    "0",
                    {},
                    "https://github.com/acme/example.git",
                ),
                DatasetRow(
                    2,
                    "acme/example",
                    "c2",
                    "tasks/second.yml",
                    "1",
                    {},
                    "https://github.com/acme/example.git",
                ),
            ]
            calls = []
            result_queue = Queue()
            dot = "digraph pdg {\n  a;\n  b;\n  a -> b;\n}\n"

            def fake_clone_repository(clone_url, destination, log):
                self.assertEqual(clone_url, "https://github.com/acme/example.git")
                tasks = destination / "tasks"
                tasks.mkdir(parents=True)
                (tasks / "first.yml").write_text("- name: first\n", encoding="utf-8")
                (tasks / "second.yml").write_text("- name: second\n", encoding="utf-8")

            def fake_run_scansible(**kwargs):
                calls.append(kwargs["target_file"].read_text(encoding="utf-8"))
                return 0, dot, ""

            def fake_load_graph(dot_path):
                graph = nx.MultiDiGraph()
                graph.add_node("a", label="source")
                graph.add_node("b", label="target")
                graph.add_edge("a", "b", label="DEF")
                return graph

            def fake_write_graphml(graph, graphml_path):
                Path(graphml_path).write_text("<graphml />\n", encoding="utf-8")

            with (
                patch(
                    "scripts.pdg_file_level_extractor.clone_repository",
                    side_effect=fake_clone_repository,
                ),
                patch("scripts.pdg_file_level_extractor.checkout_commit"),
                patch(
                    "scripts.pdg_file_level_extractor.run_scansible",
                    side_effect=fake_run_scansible,
                ),
                patch(
                    "scripts.pdg_file_level_extractor.load_and_sanitize_graph",
                    side_effect=fake_load_graph,
                ),
                patch(
                    "scripts.pdg_file_level_extractor.nx.write_graphml",
                    side_effect=fake_write_graphml,
                    create=True,
                ),
            ):
                completed = process_repository_group(
                    group_index=1,
                    total_groups=1,
                    group_key="acme/example",
                    rows=rows,
                    logs_dir=root / "logs",
                    pdg_root=root / "pdg_file_level",
                    clone_root=clone_root,
                    scansible_command="scansible",
                    timeout=10,
                    keep_dot=True,
                    min_pdg_nodes=2,
                    min_pdg_edges=1,
                    result_queue=result_queue,
                )

            results = {}
            drained = drain_result_queue(result_queue, results)
            self.assertEqual(completed, 2)
            self.assertEqual(drained, 2)
            self.assertIn('- import_tasks: "first.yml"', calls[0])
            self.assertIn('- import_tasks: "second.yml"', calls[1])
            self.assertEqual(results[1]["status"], "SUCCESS")
            self.assertEqual(results[2]["status"], "SUCCESS")
            self.assertTrue(Path(results[1]["graphml_path"]).exists())
            self.assertTrue(Path(results[2]["graphml_path"]).exists())
            self.assertFalse((clone_root / "acme__example").exists())


if __name__ == "__main__":
    unittest.main()
