import math
import unittest

from arcane import Spell, classify


class SpellTests(unittest.TestCase):
    def test_pose_mapping(self):
        self.assertEqual(classify([True, True, False, False]), "sign")
        self.assertEqual(classify([True, True, True, True]), "open")
        self.assertEqual(classify([False, False, False, False]), "other")

    def test_partial_cast_fires_once(self):
        spell = Spell()
        self.assertIsNone(spell.update(0.0, "sign", 0))
        spell.update(0.15, "sign", 0)
        spell.update(0.28, "sign", 0.5)
        self.assertIsNone(spell.update(0.35, "open", 0))
        power = spell.update(0.51, "open", 0)
        self.assertGreater(power, 0)
        self.assertLess(power, 1)
        self.assertIsNone(spell.update(0.60, "open", 0))

    def test_full_cast_survives_angle_wrap_and_brief_pose_transition(self):
        spell = Spell()
        spell.update(0.0, "sign", math.radians(170))
        spell.update(0.15, "sign", math.radians(170))
        spell.update(0.30, "sign", math.radians(-125))
        self.assertEqual(spell.power, 1)
        spell.update(0.40, "other", 0)
        spell.update(0.50, "open", 0)
        self.assertEqual(spell.update(0.65, "open", 0), 1)

    def test_long_tracking_loss_cancels_cast(self):
        spell = Spell()
        spell.update(0.0, "sign", 0)
        spell.update(0.15, "sign", 0)
        spell.update(0.30, "sign", 1.5)
        spell.update(0.90, None, 0)
        spell.update(1.0, "open", 0)
        self.assertIsNone(spell.update(1.2, "open", 0))
        self.assertFalse(spell.active)


if __name__ == "__main__":
    unittest.main()
