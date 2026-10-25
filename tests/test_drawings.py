from unittest import mock

import database
import security
from routers import drawings
from tests.test_feedback import DatabaseTestCase


class FakeCollection:
    """Stands in for the MongoDB collection: it only remembers what was inserted."""

    def __init__(self):
        self.inserted = []

    def insert_one(self, document):
        self.inserted.append(document)


FIGURE = {"x": 1.5, "y": 0.25, "z": -2.0, "forwardX": 0.0, "forwardY": 0.0, "forwardZ": 1.0}


class SaveFigureTest(DatabaseTestCase):
    """The position of the child figure comes from the game and is stored in the metadata (it used to be dropped)."""

    def setUp(self):
        super().setUp()
        self.owner = self.add_player("owner")
        self.mongo = FakeCollection()
        patcher = mock.patch.object(database, "mongo_collection", self.mongo)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, **extra):
        message = drawings.DrawingSaveMessage(owner=self.owner, gameType=2, lines=[], **extra)
        current = security.CurrentPlayer(id=self.owner, role=0)
        saved = drawings.save_drawing(message, self.db, current)
        return saved, self.mongo.inserted[-1]["metadata"]["sadChildCoordinates"]

    def test_the_figure_of_the_game_is_stored(self):
        saved, stored = self.save(sadChildCoordinates=FIGURE)
        self.assertEqual(stored, FIGURE)
        self.assertEqual(saved.metadata.sadChildCoordinates.x, 1.5)
        self.assertEqual(saved.metadata.sadChildCoordinates.forwardZ, 1.0)

    def test_without_a_figure_the_metadata_stays_empty(self):
        self.assertIsNone(self.save()[1])  # no figure shown, or a game that is older than the field
        self.assertIsNone(self.save(sadChildCoordinates=None)[1])

    def test_a_malformed_figure_is_dropped_but_the_drawing_is_still_saved(self):
        for broken in ({"x": 1.0, "y": 2.0},  # the others are missing
                       {**FIGURE, "z": "far away"},
                       {**FIGURE, "x": float("nan")},
                       {**FIGURE, "forwardY": float("inf")}):
            with self.subTest(broken=broken):
                before = len(self.mongo.inserted)
                self.assertIsNone(self.save(sadChildCoordinates=broken)[1])
                self.assertEqual(len(self.mongo.inserted), before + 1)

    def test_the_json_of_the_game_is_understood(self):
        body = ('{"owner": "%s", "gameType": 2, "lines": [], "sadChildCoordinates": '
                '{"x": 1.5, "y": 0.25, "z": -2.0, "forwardX": 0.0, "forwardY": 0.0, "forwardZ": 1.0}}' % self.owner)
        self.assertEqual(drawings.DrawingSaveMessage.model_validate_json(body).sadChildCoordinates.model_dump(), FIGURE)
        # Python's parser (and so the one of the web framework) accepts NaN, which is not valid JSON
        nan_body = body.replace('"x": 1.5', '"x": NaN')
        self.assertIsNone(drawings.DrawingSaveMessage.model_validate_json(nan_body).sadChildCoordinates)
