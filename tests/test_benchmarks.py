from traffic.benchmarks import design_stopping_sight_distance_ft


def test_stopping_sight_distance_matches_green_book():
    assert design_stopping_sight_distance_ft(30) == 200
    assert design_stopping_sight_distance_ft(25) == 155
    assert design_stopping_sight_distance_ft(35) == 250
    # Downgrades need more distance.
    assert design_stopping_sight_distance_ft(30, -3) == 205
    assert design_stopping_sight_distance_ft(30, -6) == 215
