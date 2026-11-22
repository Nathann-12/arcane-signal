from math import cos, sin, tau
import unittest

import numpy as np

from arcane import Rune, Vortex, classify, make_backdrop


def contour(turns=1, inside=False):
    points = []
    for i in range(81):
        t = i / 80
        radius = 95 * (0.78 - 0.67 * t if inside else 1)
        x = 320 + radius * cos(turns * tau * t)
        y = 240 + radius * sin(turns * tau * t)
        points.append((x, y))
    return points


class RuneTests(unittest.TestCase):
    def test_only_index_is_the_drawing_pose(self):
        self.assertEqual(classify([True, False, False, False]), "point")
        self.assertEqual(classify([True, True, False, False]), "sign")
        self.assertEqual(classify([True, True, True, True]), "open")

    def draw(self, rune, points, now):
        for position in points:
            now += 0.02
            self.assertIsNone(rune.update(now, position, True, False))
        for _ in range(10):
            now += 0.02
            self.assertIsNone(rune.update(now, points[-1], False, rune.stage in ("glyph", "ready")))
        rune.update(now + 0.01, points[-1], False, False)
        now += 0.01
        return now

    def test_ring_spiral_and_held_palm_cast_once_in_either_direction(self):
        for turns in (-0.58, 0.58):
            with self.subTest(turns=turns):
                rune = Rune()
                now = self.draw(rune, contour(0.85), 0)
                self.assertEqual(rune.stage, "glyph")
                now = self.draw(rune, contour(turns, inside=True), now)
                self.assertEqual(rune.stage, "ready")
                self.assertIsNone(rune.update(now + 0.1, (320, 240), False, True))
                self.assertIsNone(rune.update(now + 0.27, (320, 240), False, True))
                cast = rune.update(now + 0.65, (320, 240), False, True)
                self.assertIsNotNone(cast)
                self.assertGreater(len(cast.glyph), 8)
                self.assertEqual(cast.direction, 1 if turns > 0 else -1)
                self.assertIsNone(rune.update(now + 1.6, (320, 240), False, True))

    def test_zigzag_is_a_lightning_seal(self):
        rune = Rune()
        now = self.draw(rune, contour(0.85), 0)
        vertices = [(288, 194), (360, 205), (273, 260), (347, 275)]
        points = []
        for (x, y), (next_x, next_y) in zip(vertices, vertices[1:]):
            points.extend((x + (next_x - x) * t, y + (next_y - y) * t)
                          for t in np.linspace(0, 1, 20))
        self.draw(rune, points, now)
        self.assertEqual(rune.stage, "ready")
        self.assertEqual(rune.seal.kind, "lightning")

    def test_brief_pose_dropout_does_not_stop_the_pen(self):
        rune = Rune()
        rune.update(0.00, (300, 240), True, False)
        rune.update(0.05, (310, 240), False, False)
        rune.update(0.10, (320, 240), True, False)
        self.assertTrue(rune.pen_down)
        self.assertEqual(len(rune.points), 2)

    def test_incomplete_ring_and_straight_stroke_are_rejected(self):
        rune = Rune()
        now = self.draw(rune, contour(0.65), 0)
        self.assertEqual(rune.stage, "ring")
        self.assertTrue(rune.points)
        rune.clear_stroke()
        now = self.draw(rune, contour(), now)
        self.draw(rune, [(270 + i * 2, 240) for i in range(55)], now)
        self.assertEqual(rune.stage, "glyph")

    def test_moving_into_the_circle_starts_the_next_stroke_without_changing_pose(self):
        rune = Rune()
        now = self.draw(rune, contour(), 0)
        self.assertTrue(rune.outline)
        rune.update(now + 0.03, (320, 175), True, False)
        self.assertEqual(rune.points, [(320, 175)])

    def test_tracking_loss_preserves_the_stroke_and_resumes_near_the_last_point(self):
        rune = Rune()
        for i, position in enumerate(contour()[:30]):
            rune.update(i * 0.02, position, True, False)
        before = rune.points.copy()
        rune.update(0.60, None, False, False)
        rune.update(1.20, None, False, False)
        rune.update(1.25, (600, 400), True, False)
        self.assertEqual(rune.points, before)
        now = 1.25
        for position in contour()[30:]:
            now += 0.02
            rune.update(now, position, True, False)
        self.assertEqual(rune.stage, "glyph")

    def test_unrecognized_pose_waits_without_erasing_the_stroke(self):
        rune = Rune()
        for i, position in enumerate(contour()[:30]):
            rune.update(i * 0.02, position, True, False)
        before = rune.points.copy()
        rune.update(0.60, (305, 350), False, False)
        rune.update(1.20, (305, 350), False, False)
        self.assertEqual(rune.points, before)

    def test_vortex_remaps_the_live_frame(self):
        rune = Rune()
        now = self.draw(rune, contour(), 0)
        self.draw(rune, contour(1.2, inside=True), now)
        effect = Vortex(rune.seal, 0)
        backdrop = make_backdrop()
        warped = effect.warp(backdrop, 0.8)
        self.assertEqual(warped.shape, backdrop.shape)
        self.assertGreater(np.count_nonzero(warped != backdrop), 5000)


if __name__ == "__main__":
    unittest.main()
