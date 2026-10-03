"""离线验证失败存档、并行目录隔离和从旧 JD 重跑。"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main
import test as test_runner
from src.run_files import saved_run


class RunFilesTest(unittest.TestCase):
    def test_failed_initialization_and_retry_for_both_entries(self):
        for entry in (main, test_runner):
            with self.subTest(entry=entry.__name__), TemporaryDirectory() as temp:
                root = Path(temp)
                jd_file = root / 'input.txt'
                jd_file.write_text('original JD')
                with patch.object(entry, 'PROJECT_ROOT', root), \
                        patch.object(entry, 'load_config', side_effect=ValueError('bad config')):
                    self.assertEqual(entry.main(['--name', '公司__AI Engineer', '--jd-file', str(jd_file)]), 1)
                original, = (root / 'output').iterdir()
                self.assertIn('__公司__AI-Engineer__', original.name)
                self.assertEqual((original / 'jd.txt').read_text(), 'original JD')
                self.assertEqual(json.loads((original / 'run.json').read_text())['status'], 'failed')
                self.assertFalse((original / 'final_state.json').exists())
                result = {'jd_analysis': {'summary': '岗位分析'},
                          'evaluation': {'status': 'ready'},
                          'prepared_materials': '准备材料',
                          'output_markdown': 'Dear hiring manager',
                          'output_files': {'markdown_path': 'cover_letter.md'}}
                graph_builder = 'build_graph' if entry is main else 'build_test_graph'
                with patch.object(entry, 'PROJECT_ROOT', root), \
                        patch.object(entry, 'create_llm'), \
                        patch.object(entry, graph_builder), \
                        patch.object(entry, 'run_interactive', return_value=result) as run:
                    self.assertEqual(entry.main(['--name', 'retry', '--jd-file', str(original / 'jd.txt')]), 0)
                    self.assertEqual(run.call_args.args[1]['jd_text'], 'original JD')
                retry, = [p for p in (root / 'output').iterdir() if p != original]
                self.assertEqual(json.loads((retry / 'final_state.json').read_text()), result)
                record = json.loads((retry / 'run.json').read_text())
                self.assertEqual(record['status'], 'completed')
                self.assertEqual(record['jd_source'], str((original / 'jd.txt').resolve()))
                self.assertTrue((retry / 'run.log').exists())
                self.assertEqual(json.loads((original / 'run.json').read_text())['status'], 'failed')

    def test_concurrent_runs_and_names_stay_inside_root(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.txt'
            source.write_text('saved JD')
            args = SimpleNamespace(name='../' + '公司' * 100, jd_file=source)

            def run(_):
                with saved_run(args, root / 'output') as (directory, text):
                    self.assertEqual(text, 'saved JD')
                    return directory

            with ThreadPoolExecutor(max_workers=4) as pool:
                directories = list(pool.map(run, range(8)))
            self.assertEqual(len(set(directories)), 8)
            for directory in directories:
                self.assertEqual(directory.parent, root / 'output')
                self.assertLess(len(directory.name.encode()), 255)

    def test_check_creates_no_run(self):
        with TemporaryDirectory() as temp, patch.object(main, 'PROJECT_ROOT', Path(temp)), \
                patch.object(main, 'setup_logging'), patch.object(main, 'create_llm'), \
                patch.object(main, 'build_graph'):
            self.assertEqual(main.main(['--check']), 0)
            self.assertFalse((Path(temp) / 'output').exists())

    def tearDown(self):
        logger = logging.getLogger('cover_letter_agent')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


if __name__ == '__main__':
    unittest.main()
