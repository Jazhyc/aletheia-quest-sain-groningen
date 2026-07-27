from experiments.persona_belief_prompt_sweep.build_pid_control_manifest import (
    usable_manifest,
)


def test_usable_manifest_matches_original_student_filter():
    records = [
        {
            "dataset": "dev-varied-deception-a",
            "index": 1,
            "label": 0,
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:0",
        },
        {
            "dataset": "dev-varied-deception-a",
            "index": 2,
            "label": 1,
            "parse_error": True,
            "label_match": True,
            "student_target": None,
        },
        {
            "dataset": "dev-instructed-deception-a",
            "index": 3,
            "label": 1,
            "parse_error": False,
            "label_match": True,
            "student_target": "Prediction:1",
        },
    ]

    assert usable_manifest(records, "varied-deception") == [{
        "dataset": "dev-varied-deception-a",
        "index": 1,
        "label": 0,
    }]
