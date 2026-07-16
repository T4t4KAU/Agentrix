from agent_context_redundancy import analyze_cases


def test_sibling_shared_parent_redundancy() -> None:
    cases = [
        {
            "case_id": "x",
            "tokenizer": "o200k_base",
            "shared_parent_tokens": 100,
            "source_segments": [
                {"sha256": "a", "rendered_tokens": 80},
            ],
            "branches": [
                {"private_instruction": "first"},
                {"private_instruction": "second"},
            ],
        }
    ]
    report = analyze_cases(cases)
    case = report["cases"][0]
    assert case["logical_redundant_tokens"] == 100
    assert case["materialized_prompt_tokens"] > 200
    assert report["summary"]["cross_case_duplicate_segment_tokens"] == 0


def test_multiround_counts_cumulative_declared_context() -> None:
    cases = [
        {
            "case_id": "x",
            "tokenizer": "o200k_base",
            "shared_parent_tokens": 100,
            "branches": [
                {
                    "private_instruction": "first",
                    "trajectory": [
                        {"stage": "triage", "instruction": "first"},
                        {
                            "stage": "tool_followup",
                            "tool": "search",
                            "tool_observation": "found symbol",
                            "instruction": "inspect it",
                        },
                    ],
                }
            ],
        }
    ]
    case = analyze_cases(cases)["cases"][0]
    assert case["declared_model_requests"] == 2
    assert case["materialized_prompt_tokens"] > 200
    assert case["cumulative_user_tokens_materialized"] > case["declared_user_tokens"]
