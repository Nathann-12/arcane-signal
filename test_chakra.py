import unittest
from math import radians

from arcane import Chakra


class ChakraTests(unittest.TestCase):
    def frames(self, state, start, count, gap, angle=0):
        return [state.update(start + i / 30, gap, radians(angle)) for i in range(count)]

    def test_seal_charge_and_release_once(self):
        state = Chakra()
        self.frames(state, 0, 9, 1.1)
        self.assertEqual(state.stage, "charge")
        self.frames(state, 0.3, 27, 2.45)
        self.assertEqual(state.stage, "ready")
        self.assertEqual(self.frames(state, 1.2, 10, 4.2).count("release"), 1)

    def test_wide_hands_cannot_skip_the_charge(self):
        state = Chakra()
        self.frames(state, 0, 9, 1.1)
        self.frames(state, 0.3, 40, 4.1)
        self.assertEqual(state.stage, "charge")
        self.assertEqual(state.power, 0)

    def test_tracking_loss_pauses_charge_and_requires_a_fresh_release_hold(self):
        state = Chakra()
        self.frames(state, 0, 9, 1.1)
        self.frames(state, 0.3, 10, 2.5)
        power = state.power
        self.frames(state, 0.64, 15, None)
        self.assertEqual(state.power, power)
        self.frames(state, 1.14, 19, 2.5)
        self.assertEqual(state.stage, "ready")
        self.frames(state, 1.80, 4, 4.2)
        state.update(1.94, None)
        self.assertIsNone(state.update(2.60, 4.2))
        self.assertEqual(state.stage, "ready")

    def test_reacquiring_hands_does_not_count_the_missing_interval_as_charge(self):
        state = Chakra()
        self.frames(state, 0, 9, 1.1)
        self.frames(state, 0.3, 8, 2.45)
        power = state.power
        state.update(0.60, None)
        state.update(0.74, 2.45)
        self.assertEqual(state.power, power)

    def test_release_threshold_scales_with_the_charged_hand_gap(self):
        state = Chakra()
        self.frames(state, 0, 9, 1.1)
        self.frames(state, 0.3, 27, 2.35)
        self.assertEqual(state.stage, "ready")
        self.assertEqual(self.frames(state, 1.2, 10, 3.30).count("release"), 1)

    def test_turning_at_full_charge_unlocks_overdrive_once_across_the_axis_seam(self):
        state = Chakra()
        self.frames(state, 0, 9, 1.1)
        self.frames(state, 0.3, 27, 2.45, 60)
        events = []
        for i, angle in enumerate((76, 87, -81, -65, -63, -61, -60, -60, -60)):
            events.append(state.update(1.20 + i / 30, 2.45, radians(angle)))
        self.assertEqual(events.count("overdrive"), 1)
        self.assertTrue(state.overdrive)
        self.assertEqual(self.frames(state, 1.5, 10, 4.2, -60).count("release"), 1)

    def test_angle_after_tracking_loss_does_not_unlock_overdrive_by_itself(self):
        state = Chakra()
        self.frames(state, 0, 9, 1.1)
        self.frames(state, 0.3, 27, 2.45)
        self.frames(state, 1.20, 20, None)
        self.frames(state, 1.88, 20, 2.45, 56)
        self.assertFalse(state.overdrive)


if __name__ == "__main__":
    unittest.main()
