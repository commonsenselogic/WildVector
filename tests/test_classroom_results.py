from core.classroom_results import build_classroom_effect_rows


def test_classroom_results_show_clear_before_after_predictions():
    rows = build_classroom_effect_rows(
        {
            "spring migration": {
                "stopovers": {
                    "kind": "continuous",
                    "baseline": 6.9,
                    "scenario": 4.6,
                    "delta": -2.3,
                    "unit": "stops",
                },
                "corridor_choice": {
                    "kind": "classification",
                    "baseline": {"corridor 1": 0.8, "corridor 2": 0.2},
                    "scenario": {"corridor 1": 0.6, "corridor 2": 0.4},
                },
            }
        }
    )
    assert rows[0]["before"] == "6.9 stops"
    assert rows[0]["after"] == "4.6 stops"
    assert rows[0]["sentence"] == "The model predicts 2.3 fewer rest stops."
    assert rows[0]["kind"] == "quantity"
    assert rows[0]["change_short"] == "2.3 fewer"
    assert rows[0]["before_width"] == 100.0
    assert 66 < rows[0]["after_width"] < 67
    assert rows[1]["before"] == "Western Route 80%"
    assert rows[1]["after"] == "Western Route 60%"
    assert rows[1]["kind"] == "probability"
    assert rows[1]["change_short"] == "20 points less"
    assert "percentage points" in rows[1]["sentence"]
    assert rows[1]["before_width"] == 80.0
    assert rows[1]["after_width"] == 60.0
    assert "less likely" in rows[1]["sentence"]


def test_day_of_year_is_presented_as_a_date_children_can_read():
    rows = build_classroom_effect_rows(
        {
            "spring migration": {
                "arrival_date": {
                    "kind": "continuous",
                    "baseline": 139.0,
                    "scenario": 141.0,
                    "delta": 2.0,
                    "unit": "days",
                    "support": "paired-season projection",
                    "evidence_season": "spring migration",
                }
            }
        }
    )
    assert rows[0]["before"] == "May 18"
    assert rows[0]["after"] == "May 20"
    assert rows[0]["sentence"] == "Arrival is 2.0 days later."
    assert rows[0]["kind"] == "date"
    assert rows[0]["change_short"] == "2.0 days later"
    assert rows[0]["support"] == "paired-season projection"
    assert rows[0]["evidence_season"] == "spring migration"
