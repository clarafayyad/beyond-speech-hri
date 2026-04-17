import csv
import json
import os

import pytest

from interaction.experiment_logger import ExperimentLogger, FIELDNAMES


@pytest.fixture
def log_dir(tmp_path):
    """Return a temporary directory for CSV output."""
    return str(tmp_path)


@pytest.fixture
def logger(log_dir):
    """Return an ExperimentLogger that writes to a temp directory."""
    return ExperimentLogger(
        participant_id="p01",
        is_adaptive=True,
        board="01",
        key_map="5",
        log_dir=log_dir,
    )


class TestExperimentLoggerInit:
    def test_csv_created_with_header(self, logger):
        with open(logger.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == FIELDNAMES

    def test_csv_filename_contains_participant_id(self, logger):
        basename = os.path.basename(logger.csv_path)
        assert basename.startswith("experiment_p01_")
        assert basename.endswith(".csv")


class TestLogTurn:
    def _read_rows(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def test_single_turn_row(self, logger):
        logger.log_turn(
            turn=0,
            clue_word="water",
            clue_number=2,
            features={"duration": 3.5, "speech_rate": 2.1},
            confidence_level="high",
            guesses=["river", "mountain"],
            outcomes=["blue", "red"],
            score=1,
            turn_duration_s=45.123,
        )

        rows = self._read_rows(logger.csv_path)
        assert len(rows) == 1

        row = rows[0]
        assert row["participant_id"] == "p01"
        assert row["condition"] == "adaptive"
        assert row["turn"] == "0"
        assert row["clue_word"] == "water"
        assert row["clue_number"] == "2"
        assert row["confidence_level"] == "high"
        assert row["score"] == "1"
        assert row["turn_duration_s"] == "45.12"

        # JSON-encoded lists
        assert json.loads(row["guesses"]) == ["river", "mountain"]
        assert json.loads(row["outcomes"]) == ["blue", "red"]

        # Board and key_map are logged as identifiers
        assert row["board"] == "01"
        assert row["key_map"] == "5"

        # Features are JSON-encoded
        features = json.loads(row["features"])
        assert features["duration"] == 3.5

    def test_multiple_turns_appended(self, logger):
        for i in range(3):
            logger.log_turn(
                turn=i,
                clue_word=f"clue{i}",
                clue_number=i + 1,
                features=None,
                confidence_level=None,
                guesses=[f"card{i}"],
                outcomes=["blue"],
                score=1,
                turn_duration_s=10.0 + i,
            )

        rows = self._read_rows(logger.csv_path)
        assert len(rows) == 3
        assert [r["turn"] for r in rows] == ["0", "1", "2"]

    def test_none_features_produce_empty_string(self, logger):
        logger.log_turn(
            turn=0,
            clue_word="fire",
            clue_number=1,
            features=None,
            confidence_level=None,
            guesses=["castle"],
            outcomes=["neutral"],
            score=0,
            turn_duration_s=5.0,
        )

        rows = self._read_rows(logger.csv_path)
        assert rows[0]["features"] == ""
        assert rows[0]["confidence_level"] == ""

    def test_no_key_map_produces_empty_string(self, log_dir):
        logger = ExperimentLogger(
            participant_id="p02",
            is_adaptive=False,
            board=["a", "b"],
            key_map=None,
            log_dir=log_dir,
        )
        logger.log_turn(
            turn=0,
            clue_word="x",
            clue_number=1,
            features=None,
            confidence_level=None,
            guesses=["a"],
            outcomes=["blue"],
            score=1,
            turn_duration_s=1.0,
        )

        with open(logger.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["key_map"] == ""

    def test_list_board_and_dict_key_map_are_json_encoded(self, log_dir):
        logger = ExperimentLogger(
            participant_id="p03",
            is_adaptive=True,
            board=["a", "b"],
            key_map={"blue": [0], "red": [1]},
            log_dir=log_dir,
        )
        logger.log_turn(
            turn=0,
            clue_word="x",
            clue_number=1,
            features=None,
            confidence_level=None,
            guesses=["a"],
            outcomes=["blue"],
            score=1,
            turn_duration_s=1.0,
        )

        with open(logger.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert json.loads(rows[0]["board"]) == ["a", "b"]
        assert json.loads(rows[0]["key_map"]) == {"blue": [0], "red": [1]}
