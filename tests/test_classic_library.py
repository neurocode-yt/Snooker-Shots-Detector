from pathlib import Path

from tools.classic_library import classify, proposed_group, proposed_split


def test_numbered_copies_and_parts_share_a_calibration_split():
    for names in [
        ['Louis Heathcote vs Chatchapong Nasa.mp4',
         'Louis Heathcote vs Chatchapong Nasa(1).mp4',
         'Louis Heathcote vs Chatchapong Nasa(2).mp4'],
        ['Selby vs Wilson.mov', 'Selby vs Wilson p1.mov', 'Selby vs Wilson p2.mov'],
    ]:
        groups = {proposed_group(Path(name)) for name in names}
        assert len(groups) == 1
        assert len({proposed_split(group) for group in groups}) == 1


def test_archive_filenames_preserve_match_year_and_players():
    one = proposed_group(Path('Jimmy Robertson vs Thepchaiya Un-Nooh _ 2022 Championship League-f683a272.mp4'))
    another_year = proposed_group(Path('Jimmy Robertson vs Thepchaiya Un-Nooh _ 2023 Championship League-ab012345.mp4'))
    assert one != another_year
    assert one.endswith('2022_championship_league')


def test_long_promos_and_edited_clips_are_not_full_match_candidates():
    assert classify(Path('ad/match test 1.mp4'), 5000) == 'advertisement_or_promo_candidate'
    assert classify(Path('uk channel/thep vs barry.mov'), 5000) == 'edited_source_candidate'
    assert classify(Path('Mark Williams vs Andrew Higginson.mp4'), 5000) == 'match_candidate'
