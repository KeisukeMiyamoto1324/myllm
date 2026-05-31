import unittest

from src.tokenizer.training_corpus_cases import TRAINING_CORPUS_CASES


class TokenizerTrainingCorpusCasesTest(unittest.TestCase):
    def test_tokenizer_training_uses_fineweb2_edu_japanese(self) -> None:
        # ---------------------------------------------------------
        # Keep tokenizer training pointed at the requested Japanese
        # FineWeb2 Edu dataset and a single bounded train stream.
        # ---------------------------------------------------------
        corpus_cases = [
            (
                corpus_case.name,
                corpus_case.language,
                corpus_case.dataset_path,
                corpus_case.config_name,
                corpus_case.split,
                corpus_case.text_column,
                corpus_case.sample_count,
                corpus_case.max_chars,
            )
            for corpus_case in TRAINING_CORPUS_CASES
        ]

        self.assertEqual(
            corpus_cases,
            [
                (
                    "fineweb2-edu-ja",
                    "ja",
                    "hotchpotch/fineweb-2-edu-japanese",
                    "default",
                    "train",
                    "text",
                    256000,
                    4096,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
