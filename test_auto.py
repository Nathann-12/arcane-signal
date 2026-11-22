import unittest

from arcane import AutoScene, Hand, make_backdrop


def gesture(pose, x, y=245, palm_size=65, pinch_gap=1.0):
    position = (x, y)
    return Hand(pose, 0, position, position, index=position, palm_size=palm_size,
                pinch=position, pinch_gap=pinch_gap)


class AutoSceneTests(unittest.TestCase):
    def setUp(self):
        self.scene = AutoScene()
        self.frame = make_backdrop()
        self.sounds = []

    def draw(self, time, *hands):
        self.scene.draw(time, self.frame, hands, True, self.sounds.append)

    def test_stationary_point_and_brief_pose_change_do_not_steal_pulse(self):
        self.draw(0.0, gesture("point", 300))
        self.draw(0.10, gesture("point", 332))
        self.draw(0.12, gesture("sign", 335))
        self.draw(0.21, gesture("sign", 335))
        self.draw(0.33, gesture("sign", 335))
        self.assertEqual(self.scene.mode, "pulse")
        self.assertTrue(self.scene.scene.spell.active)
        self.assertEqual(self.sounds, ["summon"])

    def test_pointing_and_a_brief_pinch_never_start_a_rune(self):
        for i in range(16):
            self.draw(i / 30, gesture("point", 230 + i * 10))
        self.assertIsNone(self.scene.mode)
        self.draw(0.55, gesture("other", 390, pinch_gap=0.11))
        self.draw(0.61, gesture("other", 390, pinch_gap=0.11))
        self.draw(0.68, gesture("point", 390))
        self.assertIsNone(self.scene.mode)

    def test_rune_inks_only_after_a_held_pinch_and_pauses_when_released(self):
        for i in range(8):
            self.draw(i / 30, gesture("other", 300 + i * 6, pinch_gap=0.11))
        self.assertEqual(self.scene.mode, "runes")
        rune_scene = self.scene.scene
        self.assertEqual(rune_scene.rune.points[0], (336, 245))
        before = rune_scene.rune.points.copy()
        for i in range(8):
            self.draw(0.27 + i / 30, gesture("point", 345 + i * 7))
        self.assertEqual(rune_scene.rune.points, before)
        for i in range(9):
            self.draw(0.55 + i / 30, gesture("other", 342 + i * 3, pinch_gap=0.11))
        self.assertGreater(len(rune_scene.rune.points), len(before))
        self.draw(0.86, gesture("open", 335), gesture("open", 395))
        self.draw(0.90, gesture("open", 335), gesture("open", 395))
        self.assertIs(self.scene.scene, rune_scene)

    def test_chakra_requires_two_close_open_hands(self):
        for i in range(9):
            self.draw(i / 30, gesture("other", 400), gesture("other", 480))
        self.assertIsNone(self.scene.mode)
        for i in range(10):
            self.draw(0.35 + i / 30, gesture("open", 400), gesture("open", 480))
        self.assertEqual(self.scene.mode, "chakra")
        self.assertEqual(self.scene.scene.chakra.stage, "charge")
        self.assertEqual(self.sounds, ["summon"])

    def test_cancelled_pulse_returns_to_automatic_detection(self):
        for i in range(8):
            self.draw(i / 30, gesture("sign", 300))
        self.assertEqual(self.scene.mode, "pulse")
        self.draw(0.60)
        self.draw(0.72)
        self.draw(0.76)
        self.assertIsNone(self.scene.mode)


if __name__ == "__main__":
    unittest.main()
