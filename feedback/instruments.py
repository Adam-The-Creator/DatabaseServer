"""
The questionnaires (requirements FBK-10..FBK-14, TEC-15): which items each one has, how an answer is validated and how
it is scored. The server computes every score from the raw answers, a score sent by a client is never believed.

The item texts are not here: they live in the translations of the web app, keyed by the item ID. An answer is stored
as item ID and number, so it does not depend on the language. The version of an instrument changes whenever its
items or its scoring change.

SUS (Brooke) and UEQ (Laugwitz, Held and Schrepp) are written down here from the published instruments. Before the study
the wording and the scale assignment should be compared with the official handbooks (requirement FBK-13).
"""

from dataclasses import dataclass
from typing import Any, Optional

TARGETS = ("game", "webapp")
CATEGORIES = ("problem", "idea", "praise", "other")

# Age bands (requirement FBK-14). The limit between the first two is the one of the "children" preset of P2-01.
AGE_BANDS = ("child", "teen", "adult", "unknown")


def age_band(age: Optional[int]) -> str:
    """The age band of a player. An age of 0 is the default of a player who never filled it in."""

    if age is None or age <= 0:
        return "unknown"
    if age <= 12:
        return "child"
    return "teen" if age <= 17 else "adult"


@dataclass(frozen=True)
class Instrument:
    id: str
    version: int
    item_ids: tuple[str, ...]
    scale_min: int
    scale_max: int
    targets: tuple[str, ...]


# --- SUS: System Usability Scale (10 items, 1 = strongly disagree .. 5 = strongly agree) ---

SUS = Instrument("SUS", 1, tuple(f"sus{n}" for n in range(1, 11)), 1, 5, TARGETS)

# --- UEQ: User Experience Questionnaire (26 items, 1..7 between two opposite words) ---

UEQ = Instrument("UEQ", 1, tuple(f"ueq{n}" for n in range(1, 27)), 1, 7, TARGETS)

# The scale of every item (numbers are the item numbers 1..26 of the questionnaire)
UEQ_SCALES: dict[str, tuple[int, ...]] = {
    "attractiveness": (1, 12, 14, 16, 24, 25),
    "perspicuity": (2, 4, 13, 21),
    "efficiency": (9, 20, 22, 23),
    "dependability": (8, 11, 17, 19),
    "stimulation": (5, 6, 7, 18),
    "novelty": (3, 10, 15, 26),
}
UEQ_PRAGMATIC = ("perspicuity", "efficiency", "dependability")
UEQ_HEDONIC = ("stimulation", "novelty")
# Items whose positive word stands on the left: their answer is turned around (4 - x instead of x - 4)
UEQ_LEFT_POSITIVE = frozenset({3, 4, 5, 9, 10, 12, 17, 18, 19, 21, 23, 24, 25})

# --- SMILEY: five faces for the children (4 questions, 1 = very sad .. 5 = very happy) ---

SMILEY = Instrument("SMILEY", 1, tuple(f"smiley{n}" for n in range(1, 5)), 1, 5, ("game",))

INSTRUMENTS: dict[str, Instrument] = {i.id: i for i in (SUS, UEQ, SMILEY)}


class InvalidAnswers(ValueError):
    """The answers do not fit the instrument. The message is shown to the user."""


def definitions() -> list[dict[str, Any]]:
    """What the web app needs to build the forms: the item IDs in the order of the questionnaire, and the scale."""

    return [
        {"id": i.id, "version": i.version, "targets": list(i.targets),
         "scaleMin": i.scale_min, "scaleMax": i.scale_max, "itemIds": list(i.item_ids)}
        for i in INSTRUMENTS.values()
    ]


def get_instrument(instrument_id: str, version: int) -> Instrument:
    instrument = INSTRUMENTS.get(instrument_id)
    if instrument is None:
        raise InvalidAnswers(f"Unknown questionnaire: {instrument_id}.")
    if instrument.version != version:
        raise InvalidAnswers("This questionnaire has been updated. Please reload the page and fill it in again.")
    return instrument


def validate_answers(instrument: Instrument, answers: Any) -> dict[str, int]:
    """Exactly the items of the instrument, each an integer inside the scale (a boolean is not a number here)."""

    if not isinstance(answers, dict):
        raise InvalidAnswers("The answers are missing.")
    wanted = set(instrument.item_ids)
    if set(answers) != wanted:
        missing = wanted - set(answers)
        raise InvalidAnswers("Please answer every question." if missing else "The answers do not match the questionnaire.")
    values: dict[str, int] = {}
    for item_id in instrument.item_ids:
        value = answers[item_id]
        if isinstance(value, bool) or not isinstance(value, int) or not instrument.scale_min <= value <= instrument.scale_max:
            raise InvalidAnswers("An answer is outside the scale.")
        values[item_id] = value
    return values


# --- Scoring ---

def sus_contributions(values: dict[str, int]) -> dict[str, int]:
    """Each SUS item as 0 (worst) to 4 (best): x - 1 for the positive items (odd), 5 - x for the negative ones (even)."""

    return {item_id: (values[item_id] - 1 if n % 2 == 1 else 5 - values[item_id])
            for n, item_id in enumerate(SUS.item_ids, start=1)}


def ueq_values(values: dict[str, int]) -> dict[str, int]:
    """Each UEQ item as -3 (very negative) to +3 (very positive)."""

    return {item_id: (4 - values[item_id] if n in UEQ_LEFT_POSITIVE else values[item_id] - 4)
            for n, item_id in enumerate(UEQ.item_ids, start=1)}


def score(instrument: Instrument, values: dict[str, int]) -> dict[str, Any]:
    """SUS: the score 0..100. UEQ: the means of the six scales (-3..+3). SMILEY: the mean of the faces (1..5)."""

    if instrument.id == "SUS":
        return {"score": sum(sus_contributions(values).values()) * 2.5}
    if instrument.id == "UEQ":
        transformed = ueq_values(values)
        scales = {name: sum(transformed[f"ueq{n}"] for n in numbers) / len(numbers) for name, numbers in UEQ_SCALES.items()}
        return {
            "scales": scales,
            "pragmatic": sum(scales[s] for s in UEQ_PRAGMATIC) / len(UEQ_PRAGMATIC),
            "hedonic": sum(scales[s] for s in UEQ_HEDONIC) / len(UEQ_HEDONIC),
        }
    return {"mean": sum(values.values()) / len(values)}


def headline(instrument_id: str, scores: dict[str, Any]) -> float:
    """The one number of a submission that is followed over time: the SUS score, the mean of the UEQ scales, the smiley mean."""

    if instrument_id == "SUS":
        return scores["score"]
    if instrument_id == "UEQ":
        return sum(scores["scales"].values()) / len(scores["scales"])
    return scores["mean"]
