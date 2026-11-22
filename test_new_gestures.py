from math import radians
import unittest

import numpy as np

from arcane import AutoScene, FaceBox, Hand, PalmTwist, PalmVortex, make_backdrop, near_forehead


def hand(pose, x=480, y=179, angle=0):
    return Hand(pose, radians(angle), (x + 6, y), (x + 12, y + 90), (x, y), palm_size=70)


class GestureTests(unittest.TestCase):
    def test_palm_vortex_can_teleport_only_with_a_saved_background(self):
        for has_background in (False, True):
            auto = AutoScene()
            room = make_backdrop()
            if has_background:
                auto.background = room
            sounds = []
            for i in range(12):
                auto.draw(i / 30, room, (hand("open", angle=i * 4),), True, sounds.append)
            self.assertIsInstance(auto.scene.effect, PalmVortex)
            end = auto.scene.effect.started + auto.scene.effect.duration + 0.02
            auto.draw(end, room, (), True, sounds.append)
            self.assertEqual(auto.mode, "teleport" if has_background else None)
            self.assertEqual(sounds.count("teleport"), int(has_background))

    def test_palm_vortex_keeps_its_direction_as_the_live_picture_fades_in(self):
        for direction in (1, -1):
            twist = PalmTwist()
            for i in range(12):
                seal = twist.update(i / 30, hand("open", angle=direction * i * 4))
            effect = PalmVortex(seal, 0)
            frame = make_backdrop()
            before_fade = effect.fade_from
            self.assertGreater(before_fade, 3)
            self.assertGreater(direction * effect.rotation(effect.duration - 0.1),
                               direction * effect.rotation(before_fade))
            self.assertGreater(effect.opacity(before_fade), effect.opacity(effect.duration - 0.1))
            self.assertTrue(np.any(effect.warp(frame, before_fade) != frame))
            np.testing.assert_array_equal(effect.warp(frame, effect.duration), frame)

    def test_wrist_turn_is_required_and_carries_direction(self):
        for direction in (1, -1):
            twist = PalmTwist()
            for i in range(12):
                self.assertIsNone(twist.update(i / 30, hand("open")))
            for i in range(1, 12):
                seal = twist.update((11 + i) / 30, hand("open", angle=direction * i * 5))
                if seal:
                    break
            self.assertTrue(seal)
            self.assertEqual(seal.direction, direction)

    def test_both_fingers_must_be_close_to_the_forehead(self):
        face = FaceBox(440, 146, 100, 117)
        self.assertTrue(near_forehead(hand("sign"), face))
        self.assertFalse(near_forehead(hand("sign", y=226), face))
        self.assertFalse(near_forehead(hand("open"), face))

    def test_teleport_can_take_over_an_unfired_pulse_and_hide_the_subject(self):
        frame = make_backdrop()
        person = frame.copy()
        person[155:455, 410:580] = (158, 158, 158)
        auto = AutoScene()
        auto.background = frame
        face = FaceBox(440, 146, 100, 117)
        sounds = []
        wrist = hand("sign", x=350, y=290)
        for i in range(10):
            auto.draw(i / 30, person, (wrist,), True, sounds.append, face=face)
        self.assertEqual(auto.mode, "pulse")
        for i in range(10, 17):
            auto.draw(i / 30, person, (hand("sign"),), True, sounds.append, face=face)
        self.assertEqual(auto.mode, "teleport")
        self.assertEqual(sounds.count("teleport"), 1)
        result = auto.draw(1.18, person, (), True, sounds.append, face=face)
        np.testing.assert_array_equal(result[155:455, 410:580], frame[155:455, 410:580])
        late = auto.draw(auto.scene.started + 2.15, person, (), True, sounds.append, face=face)
        np.testing.assert_array_equal(late[155:455, 410:580], frame[155:455, 410:580])

    def test_capture_waits_until_the_room_is_clear(self):
        frame = make_backdrop()
        auto = AutoScene()
        auto.capture_background(0)
        for i in range(10):
            auto.draw(2.5 + i / 15, frame, (), True, lambda _: None, face=FaceBox(440, 146, 100, 117))
        self.assertIsNone(auto.background)
        for i in range(10):
            auto.draw(3.2 + i / 15, frame, (), True, lambda _: None)
        self.assertIsNotNone(auto.background)
        auto.reset()
        np.testing.assert_array_equal(auto.background, frame)


if __name__ == "__main__":
    unittest.main()
