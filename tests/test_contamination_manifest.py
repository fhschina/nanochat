from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_contamination_manifest import (
    COMPONENT_ID,
    CURATOR_ID,
    identify_tainted_training_ids,
    normalized_digest,
)


def test_any_eval_or_exact_vertex_taints_the_entire_component(tmp_path: Path) -> None:
    component = tmp_path / "components.parquet"
    pq.write_table(pa.table({
        CURATOR_ID: pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        COMPONENT_ID: pa.array([10, 10, 20, 20, 30, 30], type=pa.int64()),
    }), component)
    selected, group_by_id, groups = identify_tainted_training_ids(
        [component],
        eval_intervals=[(2, 3)],
        exact_training_ids=np.asarray([3], dtype=np.int64),
    )
    assert groups == {10, 20}
    assert selected == {1, 3, 4}
    assert group_by_id == {1: 10, 3: 20, 4: 20}
    assert 5 not in selected and 6 not in selected


def test_normalized_exact_match_uses_nfkc_casefold_and_whitespace() -> None:
    assert normalized_digest("  Café\nTEST ") == normalized_digest("café test")
    assert normalized_digest("ＡＢＣ") == normalized_digest("abc")
    assert normalized_digest("different") != normalized_digest("text")
