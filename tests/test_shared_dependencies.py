import ast
from pathlib import Path
import unittest


STAGE_NAMES = {
    "inference_base",
    "midtraining",
    "posttraining",
    "pretraining",
    "tokenizer",
}


class SharedDependenciesTest(unittest.TestCase):
    def test_stages_do_not_import_other_stages(self) -> None:
        # ---------------------------------------------------------
        # Keep reusable code in src.shared by rejecting direct imports
        # between training stages, tokenizer tools, and inference.
        # ---------------------------------------------------------
        source_root = Path("src")
        invalid_imports: list[str] = []

        for stage_name in sorted(STAGE_NAMES):
            for source_path in (source_root / stage_name).glob("*.py"):
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                imported_modules = [
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module is not None
                ]
                imported_modules.extend(
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                )

                for imported_module in imported_modules:
                    module_parts = imported_module.split(".")

                    if len(module_parts) < 2 or module_parts[0] != "src":
                        continue

                    imported_stage = module_parts[1]

                    if imported_stage in STAGE_NAMES and imported_stage != stage_name:
                        invalid_imports.append(f"{source_path}:{imported_module}")

        self.assertEqual(invalid_imports, [])

    def test_source_uses_shared_rich_progress(self) -> None:
        # ---------------------------------------------------------
        # Prevent direct tqdm usage from bypassing the shared Rich
        # console and creating competing terminal live displays.
        # ---------------------------------------------------------
        source_paths = Path("src").rglob("*.py")
        tqdm_imports = [
            str(source_path)
            for source_path in source_paths
            if "tqdm" in source_path.read_text(encoding="utf-8")
        ]

        self.assertEqual(tqdm_imports, [])


if __name__ == "__main__":
    unittest.main()
