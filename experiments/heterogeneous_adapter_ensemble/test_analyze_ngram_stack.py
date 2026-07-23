from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.heterogeneous_adapter_ensemble.analyze_ngram_stack import (
    append_member_features,
    attach_text,
    make_vectorizer,
)


def test_append_member_features_preserves_sparse_text_and_binary_votes() -> None:
    text = csr_matrix([[0.5, 0.0], [0.0, 1.0]])
    members = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)

    combined = append_member_features(text, members).toarray()

    assert combined.shape == (2, 5)
    assert combined[:, -3:].tolist() == members.tolist()


def test_attach_text_aligns_string_indices_and_labels() -> None:
    members = pd.DataFrame([{
        "dataset": "dataset",
        "index": "7",
        "label": 1,
        "deception": 1.0,
    }])
    text = pd.DataFrame([{
        "dataset": "dataset",
        "index": 7,
        "label": 1,
        "output_context": "Visible conversation",
    }])

    result = attach_text(members, text)

    assert result.loc[0, "text"] == "Visible conversation"
    assert "text_label" not in result


def test_vectorizer_contract_is_frozen() -> None:
    vectorizer = make_vectorizer()

    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 3
    assert vectorizer.max_features == 20_000
