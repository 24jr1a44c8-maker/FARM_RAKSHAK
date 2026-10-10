import unittest

from backend.zones import ZoneStore


class ZoneStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ZoneStore()

    def test_unscanned_zone_does_not_claim_low_risk(self):
        self.assertEqual(self.store.get(1)["risk"], "NOT_SCANNED")

    def test_counts_and_low_risk(self):
        zone = self.store.record(1, ["Healthy"] * 19 + ["Pest"])
        self.assertEqual(zone["plants_scanned"], 20)
        self.assertEqual(zone["healthy"], 19)
        self.assertEqual(zone["pest"], 1)
        self.assertEqual(zone["affected_percentage"], 5.0)
        self.assertEqual(zone["risk"], "LOW")

    def test_medium_risk(self):
        zone = self.store.record(2, ["Healthy"] * 9 + ["Disease"])
        self.assertEqual(zone["affected_percentage"], 10.0)
        self.assertEqual(zone["risk"], "MEDIUM")

    def test_high_risk(self):
        zone = self.store.record(3, ["Healthy"] * 3 + ["Pest"])
        self.assertEqual(zone["affected_percentage"], 25.0)
        self.assertEqual(zone["risk"], "HIGH")

    def test_ignores_uncertain_and_non_crop_results(self):
        zone = self.store.record(4, ["UNCERTAIN", "Non-Crop", "Healthy"])
        self.assertEqual(zone["plants_scanned"], 1)
        self.assertEqual(zone["healthy"], 1)

    def test_reset_clears_only_selected_zone(self):
        self.store.record(1, ["Pest"])
        self.store.record(2, ["Disease"])
        reset_zone = self.store.reset(1)
        self.assertEqual(reset_zone["plants_scanned"], 0)
        self.assertEqual(self.store.get(2)["disease"], 1)


if __name__ == "__main__":
    unittest.main()