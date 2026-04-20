"""Tests for prompt generation."""

from src.datasets.prompts import (
    SET_LIST_S3,
    FIRST_NAME_S3,
    generate_permuted_prompts,
)
from src.datasets.permutations import Permutation


class TestPromptGeneration:
    def test_s3_generates_six_prompts(self):
        results = generate_permuted_prompts(SET_LIST_S3)
        assert len(results) == 6

    def test_identity_is_base_prompt(self):
        results = generate_permuted_prompts(SET_LIST_S3)
        identity_result = [r for r in results if r["is_identity"]][0]
        assert identity_result["prompt"] == SET_LIST_S3.base_prompt()

    def test_all_permutations_have_correct_parity(self):
        results = generate_permuted_prompts(SET_LIST_S3)
        even_count = sum(1 for r in results if r["parity"] == 0)
        odd_count = sum(1 for r in results if r["parity"] == 1)
        assert even_count == 3
        assert odd_count == 3

    def test_permuted_prompt_swaps_entities(self):
        swap_01 = Permutation.transposition(3, 0, 1)
        prompt = SET_LIST_S3.permuted_prompt(swap_01)
        assert "Bob" in prompt.split(",")[0]  # Bob should come first
        assert "Alice" in prompt  # Alice should still be present

    def test_first_name_expected_answer(self):
        assert FIRST_NAME_S3.expected_answer_fn(["Alice", "Bob", "Carol"]) == "Alice"
        assert FIRST_NAME_S3.expected_answer_fn(["Carol", "Alice", "Bob"]) == "Carol"

    def test_all_prompts_unique(self):
        results = generate_permuted_prompts(SET_LIST_S3)
        prompts = [r["prompt"] for r in results]
        assert len(set(prompts)) == len(prompts)
