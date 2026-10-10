import json
from pathlib import Path

import pytest

from snooker_ai.segmentation.visual_tail import HandComponent,TailObservation,_confirmed_entries
from snooker_ai.types import FrameFeatures


def test_uncertain_cushion_fragment_does_not_erase_the_verified_referee_entry():
    source=json.loads((Path(__file__).parent/'fixtures/classic_hard/first-blue-tail.json').read_text())
    rows=[TailObservation(t=r['t'],full_table=r['full_table'],reset=r['reset'],
          context=FrameFeatures.model_validate(r['context']) if r['context'] else None,
          components=[HandComponent(**c) for c in r['components']]) for r in source['rows']]
    entries=_confirmed_entries(rows,44.08,.5)
    assert entries
    assert entries[0].entry_timestamp==pytest.approx(48.80)
    assert entries[0].source_entry_pts-.04==pytest.approx(48.76)


def test_native_referee_glove_can_leave_the_cushion_for_a_new_clear_footprint():
    source=json.loads((Path(__file__).parent/'fixtures/classic_hard/first-blue-native-window.json').read_text())
    rows=[TailObservation(t=r['t'],full_table=r['full_table'],reset=r['reset'],
          context=FrameFeatures.model_validate(r['context']) if r['context'] else None,
          components=[HandComponent(**c) for c in r['components']]) for r in source['rows']]
    entries=_confirmed_entries(rows,44.08,.42)
    assert entries
    assert entries[0].entry_timestamp==pytest.approx(49.48)
    assert entries[0].source_entry_pts-.04==pytest.approx(49.44)
