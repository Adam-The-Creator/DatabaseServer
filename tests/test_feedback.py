import asyncio
import json
import tempfile
import unittest
import uuid
from unittest import mock

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

import admin_account
import config
import database
import security
import throttle
from analysis.geometry import stroke_geometry
from feedback import instruments, results, store
from routers import auth
from tests.fixtures import LOCAL_SHIFT_MS, T0, legacy_stamp, make_drawing, make_line, straight

HOUR = 3600 * 1000
NOW = 1_800_000_000_000  # a fixed moment, in ms


def cid() -> str:
    return str(uuid.uuid4())


def perfect(instrument: instruments.Instrument) -> dict[str, int]:
    """The most positive answers of an instrument."""
    if instrument.id == "SUS":
        return {item: (5 if n % 2 else 1) for n, item in enumerate(instrument.item_ids, start=1)}
    if instrument.id == "UEQ":
        return {item: (1 if n in instruments.UEQ_LEFT_POSITIVE else 7) for n, item in enumerate(instrument.item_ids, start=1)}
    return {item: 5 for item in instrument.item_ids}


def neutral(instrument: instruments.Instrument) -> dict[str, int]:
    middle = (instrument.scale_min + instrument.scale_max) // 2
    return {item: middle for item in instrument.item_ids}


class InstrumentTest(unittest.TestCase):
    def test_sus_known_scores(self):
        self.assertEqual(instruments.score(instruments.SUS, neutral(instruments.SUS))["score"], 50.0)
        self.assertEqual(instruments.score(instruments.SUS, perfect(instruments.SUS))["score"], 100.0)
        worst = {item: (1 if n % 2 else 5) for n, item in enumerate(instruments.SUS.item_ids, start=1)}
        self.assertEqual(instruments.score(instruments.SUS, worst)["score"], 0.0)

    def test_sus_item_contributions_are_zero_to_four_where_four_is_best(self):
        contributions = instruments.sus_contributions(perfect(instruments.SUS))
        self.assertEqual(set(contributions.values()), {4})

    def test_ueq_scales_cover_each_item_once(self):
        numbers = sorted(n for items in instruments.UEQ_SCALES.values() for n in items)
        self.assertEqual(numbers, list(range(1, 27)))
        self.assertEqual(len(instruments.UEQ_LEFT_POSITIVE), 13)

    def test_ueq_known_scores(self):
        best = instruments.score(instruments.UEQ, perfect(instruments.UEQ))
        self.assertEqual(set(best["scales"].values()), {3.0})
        self.assertEqual((best["pragmatic"], best["hedonic"]), (3.0, 3.0))
        worst = {item: 8 - value for item, value in perfect(instruments.UEQ).items()}
        self.assertEqual(set(instruments.score(instruments.UEQ, worst)["scales"].values()), {-3.0})
        self.assertEqual(set(instruments.score(instruments.UEQ, neutral(instruments.UEQ))["scales"].values()), {0.0})

    def test_ueq_scale_mean_is_the_mean_of_its_items(self):
        answers = neutral(instruments.UEQ)
        answers["ueq2"] = 7  # perspicuity, positive on the right: +3 among four items
        answers["ueq4"] = 7  # perspicuity, positive on the left: -3
        answers["ueq13"] = 6  # +2
        scales = instruments.score(instruments.UEQ, answers)["scales"]
        self.assertEqual(scales["perspicuity"], (3 - 3 + 2 + 0) / 4)

    def test_smiley_mean(self):
        self.assertEqual(instruments.score(instruments.SMILEY, perfect(instruments.SMILEY))["mean"], 5.0)

    def test_answers_are_validated(self):
        sus = instruments.SUS
        good = neutral(sus)
        self.assertEqual(instruments.validate_answers(sus, good), good)
        for bad in (
            {k: v for k, v in good.items() if k != "sus1"},  # a missing item
            {**good, "extra": 3},  # an item that is not in the questionnaire
            {**good, "sus1": 6}, {**good, "sus1": 0},  # outside the scale
            {**good, "sus1": True}, {**good, "sus1": 2.5}, {**good, "sus1": "3"},  # not an integer
            [], None,
        ):
            with self.assertRaises(instruments.InvalidAnswers):
                instruments.validate_answers(sus, bad)

    def test_unknown_instrument_and_old_version(self):
        with self.assertRaises(instruments.InvalidAnswers):
            instruments.get_instrument("NOPE", 1)
        with self.assertRaises(instruments.InvalidAnswers):
            instruments.get_instrument("SUS", 99)

    def test_age_bands(self):
        self.assertEqual([instruments.age_band(a) for a in (None, 0, 5, 12, 13, 17, 18, 60)],
                         ["unknown", "unknown", "child", "child", "teen", "teen", "adult", "adult"])


class DatabaseTestCase(unittest.TestCase):
    """A real SQLite file with the tables of the server, in a temporary folder (never the server's own database)."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        patcher = mock.patch.object(config, "SQLITE_DB_PATH", f"{self.folder.name}/test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)
        database.verify_sqlite_schema()
        self.db = database.get_sqlite_connection()
        self.addCleanup(self.db.close)
        # Fresh in-memory protections for every test
        for name, value in (("login_guard", throttle.LoginGuard(5, 900)),
                            ("account_limiter", throttle.TokenBuckets(300, 20)),
                            ("ip_limiter", throttle.TokenBuckets(300, 20)),
                            ("auth_limiter", throttle.TokenBuckets(60, 1))):
            patcher = mock.patch.object(throttle, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def add_player(self, name: str, role: int = 0, age: int = 30) -> str:
        player_id, info_id, password_id = cid(), cid(), cid()
        self.db.execute("INSERT INTO Passwords (ID, Salt, Password) VALUES (?, 's', 'h')", (password_id,))
        self.db.execute("INSERT INTO PlayerInfo (ID, Name, Gender, Age, DominantHand) VALUES (?, ?, 'f', ?, 'right')", (info_id, name, age))
        self.db.execute(
            "INSERT INTO Players (ID, Username, PasswordID, PlayerInfoID, SignedIn, Created, Role) VALUES (?, ?, ?, ?, NULL, '2026.10.05-10:00', ?)",
            (player_id, f"user-{name}", password_id, info_id, role))
        self.db.commit()
        return player_id

    def send_questionnaire(self, player, instrument=instruments.SUS, target="game", at=NOW, answers=None, client=None):
        return store.submit_questionnaire(self.db, player, client or cid(), instrument.id, instrument.version, target,
                                          answers if answers is not None else neutral(instrument), at)

    def send_message(self, player, text="Hello", target="game", category=None, at=NOW, client=None):
        return store.submit_message(self.db, player, client or cid(), target, category, text, at)


class SubmissionTest(DatabaseTestCase):
    def test_a_questionnaire_is_stored_with_the_server_computed_score(self):
        player = self.add_player("a")
        self.send_questionnaire(player, answers=perfect(instruments.SUS))
        row = self.db.execute("SELECT Scores, Role, AgeBand, Target, InstrumentVersion FROM QuestionnaireSubmissions").fetchone()
        self.assertEqual((json.loads(row[0]), row[1:]), ({"score": 100.0}, (0, "adult", "game", 1)))

    def test_the_age_band_is_the_one_at_the_time_of_sending(self):
        child = self.add_player("kid", age=9)
        self.send_questionnaire(child, instruments.SMILEY)
        self.db.execute("UPDATE PlayerInfo SET Age = 30")
        self.db.commit()
        self.assertEqual(self.db.execute("SELECT AgeBand FROM QuestionnaireSubmissions").fetchone()[0], "child")

    def test_the_same_questionnaire_can_be_sent_again_after_the_interval_only(self):
        player = self.add_player("a")
        interval = config.FEEDBACK_QUESTIONNAIRE_INTERVAL_SECONDS * 1000
        self.send_questionnaire(player)
        with self.assertRaises(store.FeedbackError) as refused:
            self.send_questionnaire(player, at=NOW + interval - 30_000)
        self.assertEqual(refused.exception.status, 429)
        self.assertEqual(refused.exception.retry_after, 30)
        self.assertIn("30 seconds", refused.exception.detail)
        self.send_questionnaire(player, at=NOW + interval)  # now it is allowed
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM QuestionnaireSubmissions").fetchone()[0], 2)

    def test_another_instrument_or_target_is_not_blocked(self):
        psychologist = self.add_player("p", role=1)
        self.send_questionnaire(psychologist, instruments.SUS, "game")
        self.send_questionnaire(psychologist, instruments.SUS, "webapp")
        self.send_questionnaire(psychologist, instruments.UEQ, "game")

    def test_a_repeated_submission_id_is_the_same_submission(self):
        player = self.add_player("a")
        client = cid()
        first = self.send_questionnaire(player, client=client)
        again = self.send_questionnaire(player, client=client, at=NOW + 1)  # a retry: no limit, no second record
        self.assertEqual(first, again)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM QuestionnaireSubmissions").fetchone()[0], 1)

    def test_a_submission_id_of_somebody_else_is_refused(self):
        a, b = self.add_player("a"), self.add_player("b")
        client = cid()
        self.send_questionnaire(a, client=client)
        with self.assertRaises(store.FeedbackError) as refused:
            self.send_questionnaire(b, client=client)
        self.assertEqual(refused.exception.status, 409)

    def test_bad_input_is_refused_and_nothing_is_stored(self):
        player = self.add_player("a")
        for kwargs in ({"answers": {"sus1": 3}}, {"answers": {**neutral(instruments.SUS), "sus3": 9}}, {"client": "not-a-uuid"}):
            with self.assertRaises(store.FeedbackError) as refused:
                self.send_questionnaire(player, **kwargs)
            self.assertEqual(refused.exception.status, 422)
        with self.assertRaises(store.FeedbackError):
            store.submit_questionnaire(self.db, player, cid(), "SUS", 7, "game", neutral(instruments.SUS), NOW)  # old version
        with self.assertRaises(store.FeedbackError):
            self.send_questionnaire(player, target="elsewhere")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM QuestionnaireSubmissions").fetchone()[0], 0)

    def test_a_player_rates_the_game_only_a_psychologist_also_the_web_app(self):
        player, psychologist = self.add_player("a"), self.add_player("p", role=1)
        with self.assertRaises(store.FeedbackError) as refused:
            self.send_questionnaire(player, target="webapp")
        self.assertEqual(refused.exception.status, 403)
        with self.assertRaises(store.FeedbackError):
            self.send_message(player, target="webapp")
        self.send_questionnaire(psychologist, target="webapp")
        self.send_message(psychologist, target="webapp")

    def test_the_smiley_rating_is_only_for_the_game(self):
        psychologist = self.add_player("p", role=1)
        with self.assertRaises(store.FeedbackError):
            self.send_questionnaire(psychologist, instruments.SMILEY, "webapp")

    def test_an_admin_or_an_unknown_account_cannot_send(self):
        admin = self.add_player("root", role=2)
        with self.assertRaises(store.FeedbackError) as refused:
            self.send_message(admin)
        self.assertEqual(refused.exception.status, 403)
        with self.assertRaises(store.FeedbackError) as refused:
            self.send_message(cid())
        self.assertEqual(refused.exception.status, 404)

    def test_messages_are_limited_per_minute_and_per_day(self):
        player = self.add_player("a")
        self.send_message(player, at=NOW)
        with self.assertRaises(store.FeedbackError) as refused:
            self.send_message(player, at=NOW + 10_000)
        self.assertEqual((refused.exception.status, refused.exception.retry_after), (429, 50))

        spacing = config.FEEDBACK_MESSAGE_INTERVAL_SECONDS * 1000
        for n in range(1, config.FEEDBACK_MESSAGES_PER_DAY):
            self.send_message(player, at=NOW + n * spacing)
        with self.assertRaises(store.FeedbackError) as refused:
            self.send_message(player, at=NOW + config.FEEDBACK_MESSAGES_PER_DAY * spacing)
        self.assertEqual(refused.exception.status, 429)
        self.assertIn("daily limit", refused.exception.detail)
        self.send_message(player, at=NOW + 24 * HOUR + spacing)  # the first one is more than a day old now

    def test_the_cap_of_stored_messages_per_account(self):
        player = self.add_player("a")
        with mock.patch.object(config, "FEEDBACK_MAX_MESSAGES_PER_ACCOUNT", 2), \
                mock.patch.object(config, "FEEDBACK_MESSAGES_PER_DAY", 1000), \
                mock.patch.object(config, "FEEDBACK_MESSAGE_INTERVAL_SECONDS", 0):
            self.send_message(player, at=NOW)
            self.send_message(player, at=NOW + 1)
            with self.assertRaises(store.FeedbackError) as refused:
                self.send_message(player, at=NOW + 2)
        self.assertEqual(refused.exception.status, 409)

    def test_a_limit_counts_per_account(self):
        a, b = self.add_player("a"), self.add_player("b")
        self.send_message(a)
        self.send_message(b)  # b is not slowed down by a

    def test_the_message_text_is_cleaned_and_checked(self):
        player = self.add_player("a")
        self.send_message(player, "  <b>Hi</b>\r\nthere é  ", at=NOW)
        self.assertEqual(self.db.execute("SELECT Text FROM FeedbackMessages").fetchone()[0], "<b>Hi</b>\nthere é")  # stored as plain text
        for bad in ("", "   \n ", "null\x00byte", "bidi ‮ override", "x" * (config.FEEDBACK_MESSAGE_MAX_CHARS + 1), 5):
            with self.assertRaises(store.FeedbackError) as refused:
                self.send_message(player, bad, at=NOW + 1000 * HOUR)
            self.assertEqual(refused.exception.status, 422)
        with self.assertRaises(store.FeedbackError):
            self.send_message(player, "ok", category="spam", at=NOW + 2000 * HOUR)

    def test_the_pseudonym_is_stable_and_not_the_account(self):
        a, b = self.add_player("a"), self.add_player("b")
        self.send_message(a, at=NOW)
        self.send_questionnaire(a)
        self.send_message(b)
        codes = [r[0] for r in self.db.execute("SELECT Code FROM FeedbackProfiles ORDER BY Code")]
        self.assertEqual(len(codes), 2)
        self.assertRegex(codes[0], r"^P-[A-Z2-9]{6}$")
        self.assertEqual(store.pseudonym(self.db, a), store.pseudonym(self.db, a))

    def test_own_submissions_are_the_accounts_only_and_have_no_scores(self):
        a, b = self.add_player("a"), self.add_player("b")
        self.send_questionnaire(a, at=NOW)
        self.send_message(a, "mine", at=NOW + 1000)
        self.send_message(b, "not mine")
        mine = store.my_submissions(self.db, a, 25, 0)
        self.assertEqual(mine["total"], 2)
        self.assertEqual([i["kind"] for i in mine["items"]], ["message", "questionnaire"])  # newest first
        self.assertEqual(mine["items"][0]["text"], "mine")
        self.assertNotIn("scores", json.dumps(mine))
        self.assertEqual(store.my_submissions(self.db, a, 1, 1)["items"][0]["kind"], "questionnaire")

    def test_the_limit_is_decided_inside_one_transaction(self):
        # A second connection that tries to send at the same moment waits for the first and then sees its record
        player = self.add_player("a")
        other = database.get_sqlite_connection()
        self.addCleanup(other.close)
        self.send_message(player, at=NOW)
        with self.assertRaises(store.FeedbackError):
            store.submit_message(other, player, cid(), "game", None, "again", NOW + 1)


class ResultsTest(DatabaseTestCase):
    def fill(self, role=0, n=3, instrument=instruments.SUS, target="game", at=NOW, make=None, age=30):
        players = []
        for i in range(n):
            player = self.add_player(f"r{role}-{instrument.id}-{target}-{at}-{i}-{uuid.uuid4().hex[:4]}", role=role, age=age)
            players.append(player)
            answers = make(i) if make else perfect(instrument)
            self.send_questionnaire(player, instrument, target, at, answers)
        return players

    def test_a_small_group_gets_no_statistics(self):
        self.fill(n=2)
        group = results.questionnaire_results(self.db, results.Filters())["groups"][0]
        self.assertEqual((group["n"], group["enough"]), (2, False))
        self.assertNotIn("sus", group)

    def test_sus_statistics(self):
        # 50 (all neutral), 95 (the best answers, but the first item only neutral) and 100
        variants = [neutral(instruments.SUS), {**perfect(instruments.SUS), "sus1": 3}, perfect(instruments.SUS)]
        self.fill(n=3, make=lambda i: variants[i])
        group = results.questionnaire_results(self.db, results.Filters())["groups"][0]
        self.assertTrue(group["enough"])
        sus = group["sus"]
        self.assertEqual((sus["min"], sus["median"], sus["max"], sus["mean"]), (50.0, 95.0, 100.0, 81.67))
        self.assertEqual(sum(b["count"] for b in sus["histogram"]), 3)
        self.assertEqual((sus["histogram"][5]["count"], sus["histogram"][9]["count"]), (1, 2))  # 100 belongs to the last bin
        self.assertEqual(len(sus["items"]), 10)
        self.assertEqual(sus["items"][0]["mean"], 2.67)  # item 1 counts 2, 2 and 4 of 4

    def test_ueq_scales_with_a_confidence_interval(self):
        self.fill(n=3, instrument=instruments.UEQ, target="webapp", role=1,
                  make=lambda i: perfect(instruments.UEQ) if i < 2 else neutral(instruments.UEQ))
        ueq = results.questionnaire_results(self.db, results.Filters())["groups"][0]["ueq"]
        attractiveness = next(s for s in ueq["scales"] if s["id"] == "attractiveness")
        self.assertEqual(attractiveness["mean"], 2.0)  # (3 + 3 + 0) / 3
        self.assertLess(attractiveness["low"], 2.0)
        self.assertGreater(attractiveness["high"], 2.0)
        self.assertEqual(len(ueq["scales"]), 6)
        self.assertEqual(len(ueq["items"]), 26)

    def test_smiley_distribution(self):
        self.fill(n=3, instrument=instruments.SMILEY, age=8, make=lambda i: {k: 1 + i for k in instruments.SMILEY.item_ids})
        smiley = results.questionnaire_results(self.db, results.Filters())["groups"][0]["smiley"]
        self.assertEqual(smiley["mean"], 2.0)
        self.assertEqual(smiley["items"][0]["distribution"], [1, 1, 1, 0, 0])

    def test_groups_are_per_instrument_target_and_role(self):
        self.fill(role=0)
        self.fill(role=1)
        self.fill(role=1, target="webapp")
        keys = [(g["instrument"], g["target"], g["role"]) for g in results.questionnaire_results(self.db, results.Filters())["groups"]]
        self.assertEqual(keys, [("SUS", "game", 0), ("SUS", "game", 1), ("SUS", "webapp", 1)])

    def test_latest_per_account_or_all_submissions(self):
        players = self.fill(n=3)
        for player in players:  # each sends again later, with the worst answers
            self.send_questionnaire(player, at=NOW + 2 * HOUR, answers={
                item: (1 if n % 2 else 5) for n, item in enumerate(instruments.SUS.item_ids, start=1)})
        latest = results.questionnaire_results(self.db, results.Filters(counting="latest"))
        self.assertEqual((latest["groups"][0]["n"], latest["groups"][0]["sus"]["mean"]), (3, 0.0))
        every = results.questionnaire_results(self.db, results.Filters(counting="all"))
        self.assertEqual((every["groups"][0]["n"], every["groups"][0]["sus"]["mean"]), (6, 50.0))

    def test_the_trend_uses_every_submission_and_hides_small_weeks(self):
        players = self.fill(n=3, at=NOW)
        for player in players:
            self.send_questionnaire(player, at=NOW + 14 * 24 * HOUR, answers=neutral(instruments.SUS))
        self.fill(n=2, at=NOW + 28 * 24 * HOUR)  # only two that week: not shown
        trend = results.questionnaire_results(self.db, results.Filters(counting="latest"))["trend"]
        self.assertEqual([(p["n"], p["mean"]) for p in trend], [(3, 100.0), (3, 50.0)])

    def test_filters(self):
        self.fill(role=0, age=8, n=3, instrument=instruments.SMILEY)
        self.fill(role=0, n=3)
        self.fill(role=1, n=3)
        only_children = results.questionnaire_results(self.db, results.Filters(age_band="child"))
        self.assertEqual([g["instrument"] for g in only_children["groups"]], ["SMILEY"])
        self.assertEqual(len(results.questionnaire_results(self.db, results.Filters(role=1))["groups"]), 1)
        self.assertEqual(results.questionnaire_results(self.db, results.Filters(date_from="2099-01-01"))["groups"], [])
        for bad in (results.Filters(target="x"), results.Filters(counting="x"), results.Filters(date_from="soon"),
                    results.Filters(role=2), results.Filters(age_band="old")):
            with self.assertRaises(store.FeedbackError):
                results.questionnaire_results(self.db, bad)

    def test_the_results_never_name_an_account(self):
        players = self.fill(n=3)
        self.send_message(players[0], "hi")
        text = json.dumps(results.questionnaire_results(self.db, results.Filters())) + json.dumps(
            results.list_messages(self.db, limit=25, offset=0, target=None, category=None, role=None, age_band=None,
                                  date_from=None, date_to=None, read=None, q=None))
        for player in players:
            self.assertNotIn(player, text)
        self.assertNotIn("user-", text)

    def test_usage_numbers(self):
        player_with_drawing, _quiet_player = self.add_player("a"), self.add_player("b")
        psychologist, _never_signed_in = self.add_player("p", role=1), self.add_player("q", role=1)
        self.add_player("root", role=2)
        self.db.execute("UPDATE Players SET SignedIn = '2026.10.06-09:00' WHERE ID = ?", (psychologist,))
        self.db.execute("INSERT INTO DrawingMeta (ID, PlayerID, Name, Path, GameType, SessionID) VALUES ('d1', ?, 'x', 'p', 0, ?)",
                        (player_with_drawing, database.DEFAULT_SESSION_ID))
        self.db.commit()
        self.send_message(player_with_drawing)
        usage = results.usage(self.db)
        self.assertEqual(usage["registered"], {"total": 5, "players": 2, "psychologists": 2, "admins": 1})
        self.assertEqual(usage["used"], {"players": 1, "psychologists": 1})
        self.assertEqual(usage["submissions"], {"questionnaires": 0, "messages": 1, "unreadMessages": 1})
        self.assertEqual(usage["accountsPerWeek"], [{"weekStart": "2026-10-05", "players": 2, "psychologists": 2, "total": 5}])


class MessageAdminTest(DatabaseTestCase):
    def listing(self, **filters):
        base = dict(target=None, category=None, role=None, age_band=None, date_from=None, date_to=None, read=None, q=None)
        return results.list_messages(self.db, limit=25, offset=0, **{**base, **filters})

    def setUp(self):
        super().setUp()
        self.player = self.add_player("a", age=9)
        self.psychologist = self.add_player("p", role=1)
        self.send_message(self.player, "The game is fun", category="praise", at=NOW)
        self.send_message(self.psychologist, "The report page is slow", target="webapp", category="problem", at=NOW + HOUR)
        self.send_message(self.player, "More 100% colours please_", category="idea", at=NOW + 2 * HOUR)

    def test_list_filters_and_search(self):
        everything = self.listing()
        self.assertEqual((everything["total"], everything["unread"]), (3, 3))
        self.assertEqual([m["text"] for m in everything["items"]][0], "More 100% colours please_")  # newest first
        self.assertEqual(self.listing(target="webapp")["total"], 1)
        self.assertEqual(self.listing(category="idea")["total"], 1)
        self.assertEqual(self.listing(role=1)["total"], 1)
        self.assertEqual(self.listing(age_band="child")["total"], 2)
        self.assertEqual(self.listing(q="REPORT")["total"], 1)
        self.assertEqual(self.listing(q="100%")["total"], 1)  # % and _ are searched for, not wildcards
        self.assertEqual(self.listing(q="%")["total"], 1)
        self.assertEqual(self.listing(q="_")["total"], 1)
        self.assertEqual(self.listing(read="unread")["total"], 3)
        with self.assertRaises(store.FeedbackError):
            self.listing(read="maybe")

    def test_mark_read_and_delete(self):
        message_id = self.listing(q="fun")["items"][0]["id"]
        self.assertTrue(results.set_read(self.db, message_id, True)["isRead"])
        self.assertEqual((self.listing(read="read")["total"], self.listing()["unread"]), (1, 2))
        self.assertFalse(results.set_read(self.db, message_id, False)["isRead"])
        results.delete_message(self.db, message_id, "admin-id")
        self.assertEqual(self.listing()["total"], 2)
        for call in (lambda: results.set_read(self.db, message_id, True), lambda: results.delete_message(self.db, message_id, "x")):
            with self.assertRaises(store.FeedbackError) as refused:
                call()
            self.assertEqual(refused.exception.status, 404)

    def test_csv_exports_neutralise_formulas_and_keep_numbers(self):
        self.send_message(self.player, "=HYPERLINK(\"http://evil\")", at=NOW + 3 * HOUR)
        self.send_message(self.psychologist, "-2+3", at=NOW + 4 * HOUR)
        csv_text = results.messages_csv(self.db, "r", target=None, category=None, role=None, age_band=None, date_from=None,
                                        date_to=None, read=None, q=None).decode("utf-8")
        self.assertIn("'=HYPERLINK", csv_text)
        self.assertIn("'-2+3", csv_text)
        self.assertNotIn(",=HYPERLINK", csv_text)
        self.assertNotIn("user-", csv_text)

        worst = {item: 8 - value for item, value in perfect(instruments.UEQ).items()}  # every scale at -3
        for n in range(3):
            self.send_questionnaire(self.add_player(f"q{n}"), instruments.UEQ, "game", NOW, worst)
        ueq = results.questionnaires_csv(self.db, "UEQ", results.Filters(), "r").decode("utf-8")
        header, first = ueq.split("\r\n")[:2]
        self.assertTrue(header.startswith("code,role,ageBand,target,version,createdAt,attractiveness"))
        self.assertIn(",-", first)  # a negative scale mean is a number, not a text with an apostrophe
        self.assertNotIn("'-", first)

    def test_excel_dialect(self):
        self.send_questionnaire(self.player, instruments.SMILEY, "game", NOW, {k: 3 for k in instruments.SMILEY.item_ids})
        self.send_questionnaire(self.add_player("b", age=8), instruments.SMILEY, "game", NOW, {k: 4 for k in instruments.SMILEY.item_ids})
        content = results.questionnaires_csv(self.db, "SMILEY", results.Filters(), "excel")
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        self.assertIn(";", content.decode("utf-8-sig").split("\r\n")[0])
        with self.assertRaises(store.FeedbackError):
            results.questionnaires_csv(self.db, "SMILEY", results.Filters(), "tsv")


class AdminAccountTest(DatabaseTestCase):
    def sign_up(self, **kwargs):
        return auth.sign_up(auth.SignUpMessage(username="someone", password="a-password", **kwargs), self.db)

    def test_signup_registers_only_players_and_psychologists(self):
        self.assertIn("playerId", self.sign_up(role=0))
        for role in (2, -1, 7):
            with self.assertRaises(HTTPException) as refused:
                auth.sign_up(auth.SignUpMessage(username=f"x{role}", password="a-password", role=role), self.db)
            self.assertEqual(refused.exception.status_code, 400)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM Players WHERE Role = 2").fetchone()[0], 0)

    def test_create_and_update_the_admin(self):
        self.assertEqual(admin_account.create_admin(self.db, "root", "long-enough-password"), "created")
        role, signed_in = self.db.execute("SELECT Role, SignedIn FROM Players WHERE Username = 'root'").fetchone()
        self.assertEqual((role, signed_in), (2, None))
        self.assertEqual(admin_account.create_admin(self.db, "root", "another-long-password"), "updated")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM Players WHERE Username = 'root'").fetchone()[0], 1)

    def test_the_admin_can_log_in_and_the_old_password_stops_working(self):
        admin_account.create_admin(self.db, "root", "long-enough-password")
        request = Request({"type": "http", "method": "POST", "path": "/auth/login", "headers": [], "client": ("10.0.0.1", 1)})
        login = auth.login(auth.LoginMessage(username="root", password="long-enough-password"), request, self.db)
        claims = security.jwt.decode(login["accessToken"], security._SECRET, algorithms=["HS256"])
        self.assertEqual(claims["role"], 2)
        admin_account.create_admin(self.db, "root", "the-new-password")
        with self.assertRaises(HTTPException):
            auth.login(auth.LoginMessage(username="root", password="long-enough-password"), request, self.db)

    def test_weak_input_and_other_roles_are_refused(self):
        self.sign_up(role=1)
        for username, password in (("root", "short"), ("", "long-enough-password"), ("x" * 65, "long-enough-password"),
                                   ("someone", "long-enough-password")):  # the last one is a psychologist
            with self.assertRaises(admin_account.AdminAccountError):
                admin_account.create_admin(self.db, username, password)
        self.assertEqual(self.db.execute("SELECT Role FROM Players WHERE Username = 'someone'").fetchone()[0], 1)

    def test_five_failed_logins_block_the_pair_for_a_while(self):
        self.sign_up(role=0)

        def attempt(password, ip="10.0.0.1", username="someone"):
            request = Request({"type": "http", "method": "POST", "path": "/auth/login", "headers": [], "client": (ip, 1)})
            return auth.login(auth.LoginMessage(username=username, password=password), request, self.db)

        for _ in range(5):
            with self.assertRaises(HTTPException) as refused:
                attempt("wrong")
            self.assertEqual(refused.exception.status_code, 401)
        with self.assertRaises(HTTPException) as blocked:
            attempt("a-password")  # even the right password is refused while blocked
        self.assertEqual(blocked.exception.status_code, 429)
        self.assertIn("Retry-After", blocked.exception.headers)
        self.assertIn("playerId", attempt("a-password", ip="10.0.0.2"))  # from another address the user gets in
        # Unknown usernames are counted the same way, so the block does not tell which usernames exist
        for _ in range(5):
            with self.assertRaises(HTTPException):
                attempt("x", username="nobody")
        with self.assertRaises(HTTPException) as unknown:
            attempt("x", username="nobody")
        self.assertEqual(unknown.exception.status_code, 429)


def request_for(path: str, client="10.0.0.1", headers=()):
    return Request({"type": "http", "method": "GET", "path": path, "headers": list(headers), "client": (client, 1)})


def credentials(player_id: str, role: int) -> HTTPAuthorizationCredentials:
    token, _ = security.create_access_token(player_id, role)
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class AdminPermissionTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(throttle, "account_limiter", throttle.TokenBuckets(300, 20))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_an_admin_token_only_opens_the_admin_endpoints_and_the_admins_own_role_and_info(self):
        admin = credentials("adm", security.Role.ADMIN)
        for allowed in ("/admin/usage", "/admin/messages", "/players/adm/role", "/players/adm/info"):
            self.assertTrue(security.get_current_player(request_for(allowed), admin).is_admin)
        for denied in ("/sessions/", "/drawings/abc", "/players/", "/players/other/info", "/players/adm/drawings",
                       "/analysis/overview", "/feedback/mine", "/administrator", "/rooms/"):
            with self.assertRaises(HTTPException) as refused:
                security.get_current_player(request_for(denied), admin)
            self.assertEqual(refused.exception.status_code, 403, denied)

    def test_other_roles_are_not_restricted_by_that_rule(self):
        for role in (security.Role.PLAYER, security.Role.PSYCHOLOGIST):
            self.assertEqual(security.get_current_player(request_for("/sessions/"), credentials("x", role)).id, "x")

    def test_require_admin(self):
        admin = security.get_current_player(request_for("/admin/usage"), credentials("adm", security.Role.ADMIN))
        self.assertIs(security.require_admin(admin), admin)
        for role in (security.Role.PLAYER, security.Role.PSYCHOLOGIST):
            current = security.get_current_player(request_for("/admin/usage"), credentials("x", role))
            with self.assertRaises(HTTPException) as refused:
                security.require_admin(current)
            self.assertEqual(refused.exception.status_code, 403)
        with self.assertRaises(HTTPException) as no_token:
            security.get_current_player(request_for("/admin/usage"), None)
        self.assertEqual(no_token.exception.status_code, 401)

    def test_an_account_that_sends_too_many_requests_is_throttled(self):
        with mock.patch.object(throttle, "account_limiter", throttle.TokenBuckets(3, 0.001)):
            player = credentials("busy", security.Role.PLAYER)
            for _ in range(3):
                security.get_current_player(request_for("/sessions/"), player)
            with self.assertRaises(HTTPException) as refused:
                security.get_current_player(request_for("/sessions/"), player)
            self.assertEqual(refused.exception.status_code, 429)
            self.assertIn("Retry-After", refused.exception.headers)
            security.get_current_player(request_for("/sessions/"), credentials("calm", security.Role.PLAYER))


class ThrottleTest(unittest.TestCase):
    def test_token_bucket_refills_over_time(self):
        now = [0.0]
        buckets = throttle.TokenBuckets(2, 1, clock=lambda: now[0])
        self.assertEqual([buckets.take("a"), buckets.take("a")], [0.0, 0.0])
        self.assertAlmostEqual(buckets.take("a"), 1.0)  # empty: one second to the next token
        self.assertEqual(buckets.take("b"), 0.0)  # another key has its own bucket
        now[0] = 1.0
        self.assertEqual(buckets.take("a"), 0.0)
        now[0] = 100.0
        self.assertEqual([buckets.take("a"), buckets.take("a")], [0.0, 0.0])  # never more than the capacity
        self.assertGreater(buckets.take("a"), 0)

    def test_login_guard(self):
        now = [0.0]
        guard = throttle.LoginGuard(3, 60, clock=lambda: now[0])
        for _ in range(2):
            guard.failure("Ann", "1.1.1.1")
        self.assertEqual(guard.blocked_for("ann", "1.1.1.1"), 0)
        guard.failure("ANN", "1.1.1.1")  # the username is not case sensitive here
        self.assertEqual(guard.blocked_for("ann", "1.1.1.1"), 60)
        self.assertEqual(guard.blocked_for("ann", "2.2.2.2"), 0)
        now[0] = 61
        self.assertEqual(guard.blocked_for("ann", "1.1.1.1"), 0)
        guard.failure("ann", "1.1.1.1")
        guard.success("ann", "1.1.1.1")  # a good login forgets the failures
        guard.failure("ann", "1.1.1.1")
        guard.failure("ann", "1.1.1.1")
        self.assertEqual(guard.blocked_for("ann", "1.1.1.1"), 0)

    def test_human_wait(self):
        self.assertEqual([throttle.human_wait(s) for s in (0.2, 1, 45, 60, 61, 3600, 3900)],
                         ["1 second", "1 second", "45 seconds", "1 minute", "2 minutes", "1 hour", "1 hour 5 minutes"])

    def test_the_client_address_behind_a_trusted_proxy(self):
        def scope(peer, forwarded=None):
            headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
            return {"client": (peer, 1), "headers": headers}

        self.assertEqual(throttle.client_ip(scope("127.0.0.1", "203.0.113.7")), "203.0.113.7")  # ngrok on this PC
        self.assertEqual(throttle.client_ip(scope("127.0.0.1", "6.6.6.6, 203.0.113.7")), "203.0.113.7")  # the nearest one, not the faked one
        self.assertEqual(throttle.client_ip(scope("127.0.0.1", "203.0.113.7, 127.0.0.1")), "203.0.113.7")
        self.assertEqual(throttle.client_ip(scope("127.0.0.1")), "127.0.0.1")
        self.assertEqual(throttle.client_ip(scope("127.0.0.1", "not an address")), "127.0.0.1")
        self.assertEqual(throttle.client_ip(scope("198.51.100.4", "203.0.113.7")), "198.51.100.4")  # nobody else is believed

    def call(self, app, path, method="GET", headers=(), client=("203.0.113.9", 1)):
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": method, "path": path, "headers": list(headers), "client": client}
        asyncio.run(app(scope, receive, send))
        start = sent[0]
        return start["status"], dict(start["headers"]), b"".join(m.get("body", b"") for m in sent[1:])

    @staticmethod
    async def ok_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})

    def test_middleware_adds_security_headers_and_throttles_by_address(self):
        with mock.patch.object(throttle, "ip_limiter", throttle.TokenBuckets(2, 0.001)):
            app = throttle.ProtectionMiddleware(self.ok_app)
            status, headers, _ = self.call(app, "/sessions/")
            self.assertEqual(status, 200)
            self.assertEqual(headers[b"x-content-type-options"], b"nosniff")
            self.assertEqual(headers[b"x-frame-options"], b"DENY")
            self.call(app, "/sessions/")
            status, headers, body = self.call(app, "/sessions/")
            self.assertEqual(status, 429)
            self.assertIn(b"Retry-After".lower(), headers)
            self.assertIn("Too many requests", json.loads(body)["detail"])
            self.assertEqual(self.call(app, "/sessions/", client=("203.0.113.10", 1))[0], 200)  # another address

    def test_login_and_signup_have_their_own_stricter_limit(self):
        with mock.patch.object(throttle, "ip_limiter", throttle.TokenBuckets(100, 10)), \
                mock.patch.object(throttle, "auth_limiter", throttle.TokenBuckets(2, 0.001)):
            app = throttle.ProtectionMiddleware(self.ok_app)
            self.assertEqual([self.call(app, "/auth/login", "POST")[0] for _ in range(3)], [200, 200, 429])
            self.assertEqual(self.call(app, "/sessions/")[0], 200)  # everything else is not affected

    def test_feedback_answers_are_not_cached(self):
        app = throttle.ProtectionMiddleware(self.ok_app)
        self.assertEqual(self.call(app, "/feedback/mine")[1][b"cache-control"], b"no-store")
        self.assertEqual(self.call(app, "/admin/usage")[1][b"cache-control"], b"no-store")
        self.assertNotIn(b"cache-control", self.call(app, "/sessions/")[1])

    def test_a_request_body_over_the_cap_is_refused_before_it_is_read(self):
        app = throttle.ProtectionMiddleware(self.ok_app)
        too_big = lambda n: [(b"content-length", str(n).encode())]
        self.assertEqual(self.call(app, "/feedback/messages", "POST", too_big(config.FEEDBACK_MAX_BODY_BYTES))[0], 200)
        self.assertEqual(self.call(app, "/feedback/messages", "POST", too_big(config.FEEDBACK_MAX_BODY_BYTES + 1))[0], 413)
        self.assertEqual(self.call(app, "/auth/login", "POST", too_big(config.AUTH_MAX_BODY_BYTES + 1))[0], 413)
        self.assertEqual(self.call(app, "/auth/login", "POST", [(b"content-length", b"lots")])[0], 413)
        self.assertEqual(self.call(app, "/feedback/messages", "POST", [(b"transfer-encoding", b"chunked")])[0], 411)
        self.assertEqual(self.call(app, "/drawings/", "POST", too_big(10_000_000))[0], 200)  # drawings are large by design

    def test_websockets_and_lifespan_pass_through(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope["type"])

        asyncio.run(throttle.ProtectionMiddleware(app)({"type": "lifespan"}, None, None))
        self.assertEqual(reached, ["lifespan"])


class ReplayGeometryTest(unittest.TestCase):
    """The geometry says when an erased stroke was erased, so the replay can show it until then (finding on DRW-05)."""

    def erase(self, at_ms, kind=0):
        return {"eventType": kind, "invoker": "p1", "hand": 1, "timestamp": legacy_stamp(at_ms + LOCAL_SHIFT_MS)}

    def test_an_erased_stroke_carries_the_time_of_its_erasing(self):
        drawing = make_drawing([make_line("kept", straight()),
                                make_line("gone", straight(), status=0, start_ms=T0 + 2000, history=[self.erase(T0 + 5000)])])
        strokes = {s["id"]: s for s in stroke_geometry(drawing)["strokes"]}
        self.assertIsNone(strokes["kept"]["erasedAtMs"])
        self.assertEqual(strokes["gone"]["erasedAtMs"], 5000)
        self.assertEqual(stroke_geometry(drawing)["durationMs"], 5000)  # the replay lasts until the erasing

    def test_the_last_erase_or_undo_counts_and_never_before_the_stroke_ends(self):
        history = [self.erase(T0 + 3000), self.erase(T0 + 4000, kind=2), self.erase(T0 + 3500, kind=1)]  # 1 = draw, ignored
        drawing = make_drawing([make_line("a", straight(), status=0, history=history)])
        self.assertEqual(stroke_geometry(drawing)["strokes"][0]["erasedAtMs"], 4000)
        early = make_drawing([make_line("a", straight(), status=0, history=[self.erase(T0 + 100)])])  # the stroke ends at 1000
        self.assertEqual(stroke_geometry(early)["strokes"][0]["erasedAtMs"], 1000)

    def test_without_a_usable_history_the_time_is_unknown(self):
        drawing = make_drawing([make_line("a", straight(), status=0, history=[])])
        self.assertIsNone(stroke_geometry(drawing)["strokes"][0]["erasedAtMs"])
        junk = make_drawing([make_line("a", straight(), status=0, history=[{"eventType": 0, "timestamp": "garbage"}])])
        self.assertIsNone(stroke_geometry(junk)["strokes"][0]["erasedAtMs"])


if __name__ == "__main__":
    unittest.main()
