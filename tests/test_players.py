import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from routers import players
from tests.test_feedback import DatabaseTestCase


def refused(model, **fields) -> bool:
    try:
        model(**fields)
    except ValidationError:
        return True
    return False


class PlayerInfoValidationTest(unittest.TestCase):
    def test_what_the_game_sends_is_accepted(self):
        # LobbyManager.cs: the dropdowns offer these values, the age is 0 when the field is empty
        for gender in ("Male", "Female", "Other"):
            for hand in ("Left", "Right", "Both"):
                message = players.PlayerInfoUpdateMessage(name="Alex", gender=gender, age=0, dominantHand=hand)
                self.assertEqual((message.gender, message.dominantHand, message.age), (gender, hand, 0))

    def test_empty_and_missing_values_are_accepted(self):
        message = players.PlayerInfoUpdateMessage(name="", gender="", age=None, dominantHand="")
        self.assertEqual((message.name, message.gender, message.age, message.dominantHand), ("", "", None, ""))
        self.assertIsNone(players.PlayerInfoUpdateMessage().name)

    def test_extra_fields_are_ignored_like_before(self):
        self.assertEqual(players.PlayerInfoUpdateMessage(id="x", name="Alex").name, "Alex")

    def test_the_age_is_a_whole_number_from_0_to_150(self):
        for age in (0, 1, 12, 150):
            self.assertEqual(players.PlayerInfoUpdateMessage(age=age).age, age)
        for age in (-1, 151, 1000, 12.5, "ten"):
            self.assertTrue(refused(players.PlayerInfoUpdateMessage, age=age), age)

    def test_the_name_is_limited_and_has_no_control_characters(self):
        self.assertEqual(len(players.PlayerInfoUpdateMessage(name="x" * 100).name), 100)
        for name in ("x" * 101, "a\x00b", "line\nbreak", "tab\tseparated", "bell\x07", "del\x7f"):
            self.assertTrue(refused(players.PlayerInfoUpdateMessage, name=name), repr(name))
        self.assertEqual(players.PlayerInfoUpdateMessage(name="Éva Kovács-Nagy").name, "Éva Kovács-Nagy")

    def test_gender_and_hand_must_be_one_of_the_offered_values(self):
        for gender in ("male", "Female ", "x", "Non-binary"):
            self.assertTrue(refused(players.PlayerInfoUpdateMessage, gender=gender), gender)
        for hand in ("left", "Ambidextrous", "Right-handed"):
            self.assertTrue(refused(players.PlayerInfoUpdateMessage, dominantHand=hand), hand)

    def test_the_single_field_bodies_keep_their_json_and_have_the_same_limits(self):
        self.assertEqual(players.NameUpdate(**{"name": "Alex"}).name, "Alex")
        self.assertEqual(players.GenderUpdate(**{"gender": "Other"}).gender, "Other")
        self.assertEqual(players.AgeUpdate(**{"age": 9}).age, 9)
        self.assertEqual(players.DominantHandUpdate(**{"dominant_hand": "Both"}).dominant_hand, "Both")
        self.assertTrue(refused(players.NameUpdate, name="x" * 101))
        self.assertTrue(refused(players.GenderUpdate, gender="robot"))
        self.assertTrue(refused(players.AgeUpdate, age=151))
        self.assertTrue(refused(players.DominantHandUpdate, dominant_hand="none"))
        self.assertTrue(refused(players.NameUpdate))  # still required


class PlayerInfoEndpointTest(DatabaseTestCase):
    def info(self, player_id: str):
        return players.get_player_info(player_id, self.db)

    def test_the_players_info_is_updated(self):
        player = self.add_player("Alex", age=30)
        players.update_player_info(
            player, players.PlayerInfoUpdateMessage(name="Alexandra", gender="Female", age=31, dominantHand="Left"), self.db)
        info = self.info(player)
        self.assertEqual((info.name, info.gender, info.age, info.dominantHand), ("Alexandra", "Female", 31, "Left"))

    def test_the_single_field_updates_change_only_their_field(self):
        player = self.add_player("Alex", age=30)
        players.update_player_age(player, players.AgeUpdate(age=12), self.db)
        players.update_player_dominant_hand(player, players.DominantHandUpdate(dominant_hand="Both"), self.db)
        players.update_player_gender(player, players.GenderUpdate(gender="Other"), self.db)
        players.update_player_name(player, players.NameUpdate(name="Sam"), self.db)
        info = self.info(player)
        self.assertEqual((info.name, info.gender, info.age, info.dominantHand), ("Sam", "Other", 12, "Both"))

    def test_an_unknown_player_is_not_found(self):
        with self.assertRaises(HTTPException) as raised:
            players.update_player_info("nobody", players.PlayerInfoUpdateMessage(name="x"), self.db)
        self.assertEqual(raised.exception.status_code, 404)

    def test_only_the_player_themself_may_update(self):
        # the dependency of every update route is self_only (a psychologist cannot rewrite a player's data either)
        from security import self_only

        for route in players.router.routes:
            if "PUT" in getattr(route, "methods", set()):
                dependencies = [d.call for d in route.dependant.dependencies]
                self.assertIn(self_only, dependencies, route.path)


if __name__ == "__main__":
    unittest.main()
