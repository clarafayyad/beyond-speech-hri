import pytest

from interaction.utils import parse_clue


# ---------------------------------------------------------------------------
# Realistic spymaster speech (verbose / hesitant phrasing)
# ---------------------------------------------------------------------------

class TestSpymasterSpeechExamples:
    def test_contraction_hows_digit(self):
        # The original bug: STT hears "how's" instead of "house"
        assert parse_clue("how's 3") == ("hows", 3)

    def test_verbose_with_contraction(self):
        # Full spymaster utterance with filler words and a contraction
        assert parse_clue("Hmmm... let's see... I would say how's 3") == ("hows", 3)

    def test_verbose_plain_word(self):
        assert parse_clue("Hmmm... let's see... I would say ocean 2") == ("ocean", 2)

    def test_hesitation_fillers(self):
        assert parse_clue("uh um... the clue would be river 4") == ("river", 4)

    def test_ok_prefix(self):
        assert parse_clue("ok so my clue is castle 5") == ("castle", 5)

    def test_ellipsis_punctuation_stripped(self):
        assert parse_clue("fire... 3") == ("fire", 3)


# ---------------------------------------------------------------------------
# Number variations: digit, word, and homophones
# ---------------------------------------------------------------------------

class TestNumberVariations:
    def test_digit_2(self):
        assert parse_clue("water 2") == ("water", 2)

    def test_word_two(self):
        assert parse_clue("water two") == ("water", 2)

    def test_homophone_too(self):
        # "too" is recognised as 2
        assert parse_clue("water too") == ("water", 2)

    def test_homophone_to(self):
        # "to" is recognised as 2
        assert parse_clue("water to") == ("water", 2)

    def test_digit_1(self):
        assert parse_clue("fire 1") == ("fire", 1)

    def test_word_one(self):
        assert parse_clue("fire one") == ("fire", 1)

    def test_word_three(self):
        assert parse_clue("forest three") == ("forest", 3)

    def test_word_four(self):
        assert parse_clue("wind four") == ("wind", 4)

    def test_homophone_for(self):
        # "for" is recognised as 4
        assert parse_clue("wind for") == ("wind", 4)

    def test_word_five(self):
        assert parse_clue("stone five") == ("stone", 5)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse_clue("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError):
            parse_clue("   ")

    def test_no_number(self):
        with pytest.raises(ValueError):
            parse_clue("ocean")

    def test_number_too_large(self):
        with pytest.raises(ValueError):
            parse_clue("ocean 9")

    def test_number_zero(self):
        with pytest.raises(ValueError):
            parse_clue("ocean zero")

    def test_no_clue_word(self):
        # Only stopwords before the number
        with pytest.raises(ValueError):
            parse_clue("the a an 3")
