import unittest
from unittest.mock import patch

import numpy as np

from arcane import AutoScene, Hand, PurpleCharge, PurpleScene, make_backdrop


def hand(x, pose, side, y=285):
    palm = (x, y)
    return Hand(pose, 0.0, (x, y - 40), palm, index=(x, y - 45), palm_size=70,
                open_fingers=4 if pose == "open" else 1, handedness=side)


def signs(spread=145):
    return (hand(480 - spread, "open", "Left"), hand(480 + spread, "point", "Right"))


class PurpleTests(unittest.TestCase):
    def test_blue_then_red_then_confirmed_merge(self):
        charge = PurpleCharge()
        self.assertEqual(charge.update(0, signs()), "blue")
        self.assertIsNone(charge.update(0.22, signs()))
        self.assertEqual(charge.update(0.40, signs()), "red")
        self.assertIsNone(charge.update(0.50, signs(51)))
        self.assertIsNone(charge.update(0.56, ()))
        self.assertIsNone(charge.update(0.62, signs(51)))
        self.assertEqual(charge.update(0.76, signs(51)), "fuse")

    def test_combined_orb_follows_its_hand_and_waits_for_release(self):
        scene = PurpleScene(started=True)
        frame = make_backdrop()
        sounds = []
        for i in range(80):
            scene.draw(i / 30, frame, signs(51 if i >= 20 else 145), True, sounds.append)
        self.assertEqual(sounds, ["purple_form", "purple_red", "purple_fuse", "purple_ready"])
        self.assertIsNone(scene.shot_at)
        previous_x = scene.hold_position[0]
        for i in range(80, 100):
            scene.draw(i / 30, frame, (hand(445, "point", "Right"),), True, sounds.append)
        self.assertLess(scene.hold_position[0], previous_x - 30)
        scene.draw(3.36, frame, (hand(445, "open", "Left"),), True, sounds.append)
        self.assertIsNone(scene.shot_at)
        for i in range(102, 127):
            scene.draw(i / 30, frame, (hand(445, "open", "Right"),), True, sounds.append)
        self.assertIsNotNone(scene.shot_at)
        self.assertEqual(sounds.count("purple_burst"), 1)

    def test_shot_grows_toward_camera_then_explodes_on_the_firing_axis(self):
        scene = PurpleScene(started=True)
        scene.hold_position = (610, 225)
        layer = np.zeros_like(make_backdrop())
        with patch.object(scene, "draw_held") as orb, patch.object(scene, "draw_shock") as explosion:
            scene.draw_shot(layer, 0.07, 1)
            scene.draw_shot(layer, 0.32, 1.25)
            self.assertEqual([call.args[1] for call in orb.call_args_list], [(610, 225)] * 2)
            self.assertGreater(orb.call_args_list[1].args[4], orb.call_args_list[0].args[4])
            explosion.assert_not_called()
            scene.draw_shot(layer, scene.flight + 0.12, 1.5)
            self.assertEqual(explosion.call_args.args[1], (610, 225))

    def test_automatic_selection_distinguishes_purple_from_chakra(self):
        scene = AutoScene()
        room = make_backdrop()
        for i in range(10):
            scene.draw(i / 30, room, signs(), True, lambda _: None)
        self.assertEqual(scene.mode, "purple")

    def test_held_effect_preserves_pixels_away_from_the_overlay(self):
        scene = PurpleScene(started=True)
        frame = make_backdrop()
        for i in range(70):
            image = scene.draw(i / 30, frame, signs(51 if i >= 20 else 145), True, lambda _: None)
        np.testing.assert_array_equal(image[125:240, 12:72], frame[125:240, 12:72])


if __name__ == "__main__":
    unittest.main()
