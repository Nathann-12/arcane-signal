import argparse
from dataclasses import dataclass
from math import atan2, cos, exp, hypot, pi, radians, sin, tau
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic
from urllib.request import urlopen
import wave

import cv2
import numpy as np


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
WINDOW = "Arcane Signal"


# Shared drawing and the pulse effect

WIDTH, HEIGHT = 960, 540
ICE = (255, 205, 105)
GOLD = (90, 186, 255)
WHITE = (255, 250, 232)
MUTED = (154, 135, 120)


def point(x, y):
    return round(x), round(y)


def polar(center, radius, angle):
    return point(center[0] + radius * cos(angle), center[1] + radius * sin(angle))


def tint(color, factor):
    return tuple(round(channel * factor) for channel in color)


def line(layer, start, end, color, width=1):
    cv2.line(layer, start, end, color, width, cv2.LINE_AA)


def ring(layer, center, radius, color, width=1):
    cv2.circle(layer, center, max(1, round(radius)), color, width, cv2.LINE_AA)


def arc(layer, center, radius, start, end, color, width=1):
    if end > start:
        size = max(1, round(radius))
        cv2.ellipse(layer, center, (size, size), 0, start, end, color, width, cv2.LINE_AA)


def polygon(layer, corners, color, fill=False):
    vertices = np.asarray(corners, dtype=np.int32)
    if fill:
        cv2.fillConvexPoly(layer, vertices, color, cv2.LINE_AA)
    else:
        cv2.polylines(layer, [vertices], True, color, 1, cv2.LINE_AA)


def sigil(layer, center, power, turn, now):
    ready = power >= 1
    color = GOLD if ready else ICE
    radius = 56 + 29 * power
    spin = turn * 1.8 + now * 0.4
    ring(layer, center, radius, color, 2)
    ring(layer, center, radius - 9, tint(color, 0.48))
    ring(layer, center, radius * 0.48, tint(color, 0.46))
    for side in range(6):
        angle = spin + side * tau / 6
        diamond = [
            polar(center, radius + 19, angle),
            polar(center, radius + 11, angle + 0.045),
            polar(center, radius + 4, angle),
            polar(center, radius + 11, angle - 0.045),
        ]
        polygon(layer, diamond, color, side % 2 == 0)
        arc(layer, center, radius + 13, angle * 180 / pi + 9, angle * 180 / pi + 49,
            tint(color, 0.5))
    for tick in range(48):
        angle = -spin * 0.6 + tick * tau / 48
        length = 7 if tick % 4 == 0 else 3
        line(layer, polar(center, radius - 13, angle),
             polar(center, radius - 13 - length, angle), tint(color, 0.54))
    for offset in (0, pi / 3):
        triangle = [polar(center, radius * 0.69, spin + offset + j * tau / 3)
                    for j in range(3)]
        polygon(layer, triangle, color)
    for index in range(16):
        angle = index * 2.4 - now * (0.5 + index % 3 * 0.13)
        distance = radius + 20 + 12 * sin(index + now * 2)
        ring(layer, polar(center, distance, angle), 1 + (index % 4 == 0),
             tint(color, 0.45 + 0.40 * sin(index + now) ** 2), -1)
    arc(layer, center, radius + 6, -90, -90 + 360 * power, WHITE, 2)
    ring(layer, center, 3 + 3 * power, WHITE, -1)


@dataclass
class Cast:
    start: tuple
    end: tuple
    time: float
    power: float

    flight = 0.38
    lifetime = 1.65

    @property
    def full(self):
        return self.power >= 1

    @property
    def color(self):
        return GOLD if self.full else ICE

    def draw(self, layer, now):
        age = now - self.time
        if age < self.flight:
            self.draw_wave(layer, age / self.flight)
        else:
            self.draw_impact(layer, age - self.flight)

    def draw_wave(self, layer, progress):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = max(1, hypot(dx, dy))
        across = (-dy / length, dx / length)
        head = point(self.start[0] + dx * progress, self.start[1] + dy * progress)
        ring(layer, self.start, 17 + 105 * progress, tint(self.color, 1 - progress), 2)
        for trail in range(5):
            along = max(0, progress - trail * 0.065)
            center = point(self.start[0] + dx * along, self.start[1] + dy * along)
            spread = (13 + self.power * 19 + trail * 8) * min(progress * 12, 1)
            vertices = [
                point(center[0] + across[0] * spread * offset - dx / length * offset ** 2 * 14,
                      center[1] + across[1] * spread * offset - dy / length * offset ** 2 * 14)
                for offset in np.linspace(-1, 1, 17)
            ]
            cv2.polylines(layer, [np.asarray(vertices, dtype=np.int32)], False,
                          tint(self.color, 1 - trail * 0.16), 3 if trail == 0 else 1, cv2.LINE_AA)
        line(layer, point(head[0] - dx / length * 50, head[1] - dy / length * 50), head, self.color, 2)
        ring(layer, head, 5 + self.power * 4, WHITE, -1)

    def draw_impact(self, layer, age):
        fade = max(0, 1 - age / (0.85 if self.full else 0.48))
        if fade == 0:
            return
        for index in range(3):
            radius = 18 + age * (222 - index * 53) + index * 7
            arc(layer, self.end, radius, index * 91 + age * 130,
                index * 91 + age * 130 + 294, tint(self.color, fade * (1 - index * 0.19)), 2)
        rng = Random(47)
        for index in range(54 if self.full else 20):
            angle = rng.uniform(0, tau)
            speed = rng.uniform(75, 285)
            distance = 12 + speed * age
            x, y = polar(self.end, distance, angle)
            y += round(80 * age ** 2)
            p = (x, y)
            line(layer, polar(p, 8 + 23 * fade, angle + pi), p, tint(self.color, fade))
            if index % 4 == 0 and self.full:
                vertices = [polar(p, 2 + 7 * fade, angle + age * 7 + k * tau / 3) for k in range(3)]
                polygon(layer, vertices, tint(WHITE, fade), True)
        if age < 0.13:
            ring(layer, self.end, 12 + 77 * age, tint(WHITE, 1 - age / 0.13), -1)


def orb_position(now, cast):
    x, y = 748, 250 + 12 * sin(now * 2.3)
    if cast:
        impact_age = now - cast.time - cast.flight
        if impact_age < 0:
            return cast.end
        if cast.full and impact_age < 0.90:
            return None
        if not cast.full:
            dx, dy = cast.end[0] - cast.start[0], cast.end[1] - cast.start[1]
            offset = 100 * sin(impact_age * 7) * exp(-4 * impact_age) / max(1, hypot(dx, dy))
            x += dx * offset
            y += dy * offset
    return point(x, y)


def orb(layer, center, now):
    ring(layer, center, 27, (77, 44, 34), -1)
    ring(layer, center, 28, ICE, 2)
    arc(layer, center, 40, now * 65, now * 65 + 190, tint(ICE, 0.65))
    arc(layer, center, 45, -now * 55 + 150, -now * 55 + 242, tint(ICE, 0.4))
    corners = [polar(center, 20, now * 0.85 + index * tau / 4) for index in range(4)]
    polygon(layer, corners, ICE)
    for corner in corners:
        line(layer, corner, center, tint(ICE, 0.6))
    ring(layer, center, 4, WHITE, -1)


def make_backdrop():
    rows, columns = np.ogrid[:HEIGHT, :WIDTH]
    glow = np.exp(-(((columns - 430) / 520) ** 2 + ((rows - 265) / 280) ** 2))
    image = np.stack((16 + glow * 29, 12 + glow * 16, 13 + glow * 18), axis=-1).astype(np.uint8)
    rng = Random(814)
    for _ in range(140):
        position = (rng.randrange(WIDTH), rng.randrange(HEIGHT))
        ring(image, position, 1, tint(ICE, rng.uniform(0.12, 0.37)), -1)
    for y in range(360, HEIGHT, 21):
        line(image, (0, y), (WIDTH, y), (42, 29, 28))
    for x in range(-600, WIDTH + 600, 75):
        line(image, (WIDTH // 2, 340), (x, HEIGHT), (42, 29, 28))
    return image


def composite(frame, layer):
    small = cv2.resize(layer, (WIDTH // 3, HEIGHT // 3), interpolation=cv2.INTER_AREA)
    halo = cv2.GaussianBlur(small, (0, 0), 4)
    halo = cv2.resize(halo, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    halo = cv2.addWeighted(halo, 1.7, halo, 0, 0)
    return cv2.add(cv2.add(frame, halo), layer)


def label(frame, text, position, scale, color=WHITE, width=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, width, cv2.LINE_AA)


def display(frame, spell, status, preview):
    cv2.rectangle(frame, (19, 19), (WIDTH - 20, HEIGHT - 20), (66, 61, 60), 1, cv2.LINE_AA)
    for x, direction in ((19, 1), (WIDTH - 20, -1)):
        for y, down in ((19, 1), (HEIGHT - 20, -1)):
            line(frame, (x, y), (x + 18 * direction, y), ICE)
            line(frame, (x, y), (x, y + 18 * down), ICE)
    cv2.rectangle(frame, (29, 31), (309, 106), (22, 17, 17), -1)
    label(frame, "PULSE", (43, 62), 0.84, WHITE, 2)
    label(frame, "EFFECTS PREVIEW" if preview else "LIVE CAMERA", (44, 87), 0.44, ICE)
    cv2.rectangle(frame, (29, 462), (WIDTH - 30, 507), (22, 17, 17), -1)
    label(frame, status, (43, 490), 0.5)
    if spell.active:
        color = GOLD if spell.power >= 1 else ICE
        cv2.rectangle(frame, (557, 480), (731, 490), (56, 49, 46), -1)
        if spell.power > 0:
            cv2.rectangle(frame, (557, 480), (557 + round(spell.power * 174), 490), color, -1)
        label(frame, f"{spell.power:.0%}", (742, 490), 0.45, color)
    label(frame, "ESC  Close", (822, 490), 0.43, MUTED)


# Hand landmarks and sound

@dataclass
class Hand:
    pose: str
    angle: float
    fingers: tuple
    palm: tuple
    index: tuple = (0, 0)
    palm_size: float = 0.0
    open_fingers: int = 0
    pinch: tuple | None = None
    pinch_gap: float = 1.0
    handedness: str | None = None

    @classmethod
    def from_landmarks(cls, landmarks):
        coordinates = np.array([(p.x * WIDTH, p.y * HEIGHT, p.z * WIDTH) for p in landmarks])
        extended = []
        for base in (5, 9, 13, 17):
            root, middle, _, tip = coordinates[base:base + 4]
            first, whole = middle - root, tip - root
            length = np.linalg.norm(whole)
            proximal = np.linalg.norm(first)
            straightness = np.dot(first, whole) / max(proximal * length, 1e-6)
            reach = length / max(proximal, 1e-6)
            extended.append(straightness > 0.80 and reach > 1.62)
        palm_axis = coordinates[9] - coordinates[0]
        palm_size = float(np.linalg.norm(palm_axis[:2]))
        fingertips = coordinates[[8, 12], :2].mean(axis=0)
        palm = coordinates[[0, 5, 9, 13, 17], :2].mean(axis=0)
        pinch = coordinates[[4, 8], :2].mean(axis=0)
        tip_gap = np.linalg.norm(coordinates[4] - coordinates[8])
        pinch_gap = 1.0
        if np.linalg.norm(pinch - palm) >= 0.43 * palm_size:
            pinch_gap = float(tip_gap / max(np.linalg.norm(palm_axis), 1e-6))
        pose = classify(extended)
        return cls(pose, atan2(palm_axis[1], palm_axis[0]),
                   tuple(fingertips), tuple(palm), tuple(coordinates[8, :2]), palm_size,
                   sum(map(int, extended)), tuple(pinch), pinch_gap)


def load_model(filename="hand_landmarker.task", url=MODEL_URL):
    cache = Path(__file__).resolve().with_name(".cache")
    cache.mkdir(exist_ok=True)
    target = cache / filename
    if not target.exists():
        print(f"Downloading {filename}...", flush=True)
        with urlopen(url, timeout=25) as response:
            model = response.read()
        with NamedTemporaryFile(dir=cache, delete=False) as download:
            temporary = Path(download.name)
            download.write(model)
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target.read_bytes()


def make_tracker(num_hands=1):
    import mediapipe as mp

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=load_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.55,
    )
    return vision.HandLandmarker.create_from_options(options)


def track_hands(tracker, frame, timestamp):
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = tracker.detect_for_video(image, timestamp)
    hands = [Hand.from_landmarks(landmarks) for landmarks in result.hand_landmarks]
    for hand, categories in zip(hands, result.handedness):
        if categories:
            hand.handedness = categories[0].category_name
    return hands


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float


def make_face_detector():
    import mediapipe as mp

    vision = mp.tasks.vision
    options = vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=load_model("blaze_face_short_range.tflite", FACE_URL)),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.52,
    )
    return vision.FaceDetector.create_from_options(options)


def track_face(detector, frame, timestamp):
    import mediapipe as mp

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = detector.detect_for_video(image, timestamp).detections
    if not detections:
        return None
    box = max((detection.bounding_box for detection in detections),
              key=lambda bounds: bounds.width * bounds.height)
    return FaceBox(box.origin_x, box.origin_y, box.width, box.height)


def camera_frame(raw):
    height, width = raw.shape[:2]
    scale = max(WIDTH / width, HEIGHT / height)
    resized = cv2.resize(raw, (round(width * scale), round(height * scale)))
    x = (resized.shape[1] - WIDTH) // 2
    y = (resized.shape[0] - HEIGHT) // 2
    return cv2.flip(resized[y:y + HEIGHT, x:x + WIDTH], 1)


def prepare_sounds(directory):
    sample_rate = 22050
    rng = np.random.default_rng(41)
    sounds = {
        "summon": (0.22, 230, 1050, 0.04),
        "charge_1": (0.19, 390, 830, 0.03),
        "charge_2": (0.19, 620, 1490, 0.03),
        "ready": (0.30, 940, 1280, 0.04),
        "cast": (0.39, 340, 55, 0.30),
        "hit": (0.27, 160, 64, 0.22),
        "shatter": (0.64, 740, 60, 0.48),
        "vortex": (1.24, 72, 510, 0.28),
        "rupture": (1.05, 178, 31, 0.64),
        "lightning_charge": (0.65, 220, 1300, 0.52),
        "thunder": (1.16, 104, 29, 0.96),
        "chakra_ready": (0.55, 510, 1420, 0.03),
        "chakra_release": (1.27, 155, 30, 0.88),
        "chakra_overdrive": (0.58, 370, 1140, 0.12),
        "chakra_collapse": (0.43, 820, 48, 0.23),
        "chakra_rift": (1.38, 126, 33, 0.75),
        "purple_form": (0.67, 120, 740, 0.13),
        "purple_red": (0.52, 790, 117, 0.30),
        "purple_fuse": (0.72, 820, 42, 0.34),
        "purple_ready": (0.48, 396, 890, 0.12),
        "purple_burst": (1.86, 143, 30, 0.87),
        "teleport": (0.42, 1960, 320, 0.40),
    }
    for name, (duration, start, end, noise) in sounds.items():
        t = np.arange(round(sample_rate * duration)) / sample_rate
        phase = tau * (start * t + (end - start) / (2 * duration) * t ** 2)
        envelope = np.minimum(1, t / 0.025) * np.exp(-4.1 * t / duration)
        if name == "vortex":
            envelope = np.minimum(1, t / 0.34) * (0.25 + t / duration) * np.minimum(1, (duration - t) / 0.06)
        signal = 0.55 * np.sin(phase) + 0.25 * np.sin(phase * 1.5) + noise * rng.uniform(-1, 1, t.size)
        if name == "thunder":
            signal *= 0.57 + 0.43 * np.cos(tau * 17 * t)
        if name in ("chakra_release", "chakra_rift", "purple_burst"):
            signal += 0.24 * np.sin(tau * (42 * t + 2 * np.sin(t * 16)))
        if name == "purple_burst":
            signal += 0.36 * np.sin(tau * (65 * t - 11 * t ** 2)) * np.exp(-2.2 * t)
        if name == "chakra_rift":
            signal *= 0.8 + 0.2 * np.cos(tau * 13 * t)
        if name == "teleport":
            envelope = np.minimum(1, t / 0.005) * np.exp(-14 * t)
            signal += 0.32 * np.sin(phase * 2.04 + 2.6 * np.sin(tau * 91 * t))
            signal += 0.35 * rng.uniform(-1, 1, t.size) * np.exp(-t * 65)
            clip = Path(__file__).with_name("assets") / "teleport.wav"
            if clip.exists():
                with wave.open(str(clip)) as source:
                    if (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, sample_rate):
                        clip_samples = np.frombuffer(source.readframes(t.size), dtype="<i2") / 32768
                        signal[:clip_samples.size] += clip_samples * 0.55
        signal = np.clip(signal, -1.32, 1.32)
        samples = (signal * envelope * 24500).astype("<i2")
        with wave.open(str(Path(directory) / f"{name}.wav"), "wb") as file:
            file.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            file.writeframes(samples.tobytes())
    original = Path(__file__).with_name("assets") / "teleport-original.wav"
    if original.is_file():
        try:
            with wave.open(str(original)) as source:
                if source.getnframes() and source.getsampwidth() in (1, 2):
                    (Path(directory) / "teleport.wav").write_bytes(original.read_bytes())
        except (OSError, EOFError, wave.Error):
            pass


# Two-finger pulse

def classify(extended):
    if all(extended):
        return "open"
    if list(extended) == [True, True, False, False]:
        return "sign"
    if list(extended) == [True, False, False, False]:
        return "point"
    return "other"


class Spell:
    def __init__(self):
        self.active = False
        self.power = 0.0
        self.angle = 0.0
        self.turn = 0.0
        self.last_sign = 0.0
        self.pose = None
        self.pose_since = 0.0

    def update(self, now, pose, angle):
        if pose != self.pose:
            self.pose = pose
            self.pose_since = now
        steady = now - self.pose_since >= 0.12

        if pose == "sign":
            if self.active:
                if now - self.last_sign <= 0.20:
                    delta = angle - self.angle
                    self.turn += atan2(sin(delta), cos(delta))
                    self.power = max(self.power, min(abs(self.turn) / radians(60), 1))
            elif steady:
                self.active = True
                self.turn = self.power = 0.0
            self.angle = angle
            self.last_sign = now
        elif self.active:
            if now - self.last_sign > 0.45:
                self.active = False
            elif pose == "open" and steady:
                self.active = False
                return self.power
        return None


class PulseScene:
    def __init__(self):
        self.spell = Spell()
        self.center = None
        self.cast = None
        self.impact_played = False
        self.charged_at = -1.0

    def draw(self, now, frame, hand, preview, play):
        layer = np.zeros_like(frame)
        was_active = self.spell.active
        past_power = self.spell.power
        if self.cast and now - self.cast.time >= self.cast.lifetime:
            self.cast = None
        pose = hand.pose if hand and self.cast is None else None
        power = self.spell.update(now, pose, hand.angle if hand else 0)
        if power is not None and hand:
            self.cast = Cast(point(*hand.palm), orb_position(now, None), now, power)
            self.impact_played = False
            play("cast")

        target = orb_position(now, self.cast)
        if target is not None:
            orb(layer, target, now)
        if self.cast:
            self.cast.draw(layer, now)
            if now - self.cast.time >= self.cast.flight and not self.impact_played:
                play("shatter" if self.cast.full else "hit")
                self.impact_played = True

        status = "Hold up two fingers to summon"
        if self.spell.active:
            if hand and pose == "sign":
                x, y = hand.fingers
                if was_active and self.center:
                    x = self.center[0] * 0.53 + x * 0.47
                    y = self.center[1] * 0.53 + y * 0.47
                self.center = (x, y)
            if self.center:
                sigil(layer, point(*self.center), self.spell.power, self.spell.turn, now)
            status = "Tilt your hand to charge  /  Open palm to cast"
            if not was_active:
                play("summon")
            elif past_power < 1 <= self.spell.power:
                self.charged_at = now
                play("ready")
            elif past_power < 0.67 <= self.spell.power:
                play("charge_2")
            elif past_power < 0.33 <= self.spell.power:
                play("charge_1")
            if self.spell.power >= 1:
                status = "FULL CHARGE  /  Open palm to cast"
                flash = (now - self.charged_at) / 0.24
                if self.center and 0 <= flash < 1:
                    ring(layer, point(*self.center), 88 + 46 * flash,
                         tint(WHITE, 1 - flash), 3)
        elif self.cast:
            if now - self.cast.time < self.cast.flight:
                status = "PULSE CAST"
            else:
                status = "TARGET SHATTERED" if self.cast.full else "TARGET KNOCKED BACK"
        self.center = self.center if self.spell.active else None
        output = composite(frame, layer)
        display(output, self.spell, status, preview)
        return output


def demo_hand(now):
    phase = now % 4.4
    if phase < 0.36 or phase > 2.45:
        return None
    full = int(now // 4.4) % 2 == 0
    end_turn = radians(64 if full else 31)
    bend = end_turn * min(1, max(0, (phase - 0.64) / 1.05))
    pose = "sign" if phase < 1.97 else "open"
    return Hand(pose, bend, (322 + 35 * sin(phase * 1.3), 239), (345, 286))


# Drawn vortex and lightning seals

@dataclass
class Seal:
    center: tuple
    radius: float
    outline: list
    glyph: list
    kind: str
    direction: int


def winding(offsets):
    angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    turn = angles[-1] - angles[0]
    travel = np.abs(np.diff(angles)).sum()
    return turn, travel


def ring_shape(points):
    if len(points) < 12:
        return None
    xy = np.array(points)
    low, high = xy.min(axis=0), xy.max(axis=0)
    center = (high + low) / 2
    offsets = xy - center
    radii = np.linalg.norm(offsets, axis=1)
    radius = float(np.median(radii))
    turn, travel = winding(offsets)
    if (radius < 32 or min(high - low) < 0.58 * max(high - low)
            or np.linalg.norm(xy[0] - xy[-1]) > radius * 1.2
            or radii.std() > radius * 0.28
            or abs(turn) < 0.79 * tau or abs(turn) < 0.75 * travel):
        return None
    return tuple(map(float, center)), radius


def spiral_direction(points, center, radius):
    if len(points) < 9:
        return 0
    offsets = np.array(points) - center
    radii = np.linalg.norm(offsets, axis=1)
    if (np.percentile(radii, 90) > radius * 1.15
            or min(radii[0], radii[-1]) > radius * 0.53
            or abs(radii[-1] - radii[0]) < radius * 0.19):
        return 0
    around_center = offsets[radii > radius * 0.13]
    if len(around_center) < 7:
        return 0
    turn, travel = winding(around_center)
    if abs(turn) < 0.42 * tau or abs(turn) < 0.73 * travel:
        return 0
    return 1 if turn > 0 else -1


def is_zigzag(points, center, radius):
    if len(points) < 9:
        return False
    vertices = np.array(points, dtype=np.float32)
    if np.percentile(np.linalg.norm(vertices - center, axis=1), 90) > radius * 1.18:
        return False
    corners = cv2.approxPolyDP(vertices, radius * 0.10, False).reshape(-1, 2)
    if len(corners) < 4:
        return False
    turns = np.diff(corners, axis=0)
    lengths = np.linalg.norm(turns, axis=1)
    long_enough = (lengths[:-1] >= radius * 0.22) & (lengths[1:] >= radius * 0.22)
    sharp = (turns[:-1] * turns[1:]).sum(axis=1) < 0.14 * lengths[:-1] * lengths[1:]
    return bool(np.any((long_enough & sharp)[:-1] & (long_enough & sharp)[1:]))


class Rune:
    def __init__(self):
        self.stage = "ring"
        self.points = []
        self.outline = []
        self.seal = None
        self.center = self.radius = None
        self.paused = False
        self.last_motion = 0.0
        self.open_since = None
        self.ready_at = -10.0

    @property
    def pen_down(self):
        return bool(self.points)

    def clear_stroke(self):
        self.points = []
        self.paused = False

    def update(self, now, position, inking, open_palm):
        if self.stage == "cast":
            return None
        if not open_palm or position is None:
            self.open_since = None
        elif self.stage == "glyph" and self.lock_glyph(now):
            self.open_since = now
        if self.stage == "ready":
            if open_palm and position is not None:
                if self.open_since is None:
                    self.open_since = now
                elif now - self.open_since >= 0.35:
                    self.stage = "cast"
                    return self.seal
            return None
        if position is None or not inking or open_palm:
            self.paused = self.pen_down
            return None
        if not self.pen_down:
            if self.stage == "ring" or hypot(position[0] - self.center[0], position[1] - self.center[1]) <= self.radius * 0.86:
                self.points = [position]
                self.last_motion = now
            return None
        x, y = self.points[-1]
        dx, dy = position[0] - x, position[1] - y
        distance = hypot(dx, dy)
        if distance > 78:
            self.paused = True
            return None
        self.paused = False
        if distance >= 3.5:
            if len(self.points) >= 1000:
                self.points = self.points[::2]
            self.points.append((x + 0.84 * dx, y + 0.84 * dy))
            self.last_motion = now
        if self.stage == "ring":
            shape = ring_shape(self.points)
            if shape:
                self.lock_ring(shape)
        elif now - self.last_motion >= 0.40:
            self.lock_glyph(now)
        return None

    def lock_ring(self, shape):
        self.center, self.radius = shape
        self.outline, self.points = self.points, []
        self.stage = "glyph"

    def lock_glyph(self, now):
        if len(self.points) < 9:
            return False
        direction = spiral_direction(self.points, self.center, self.radius)
        if is_zigzag(self.points, self.center, self.radius):
            kind = "lightning"
        elif direction:
            kind = "vortex"
        else:
            return False
        self.seal = Seal(self.center, self.radius, self.outline, self.points, kind, direction)
        self.clear_stroke()
        self.stage = "ready"
        self.ready_at = now
        return True


INK = (253, 120, 224)
EMBER = (87, 173, 255)
FROST = (255, 237, 85)
VIOLET = (255, 121, 191)
HALF_GRID = np.mgrid[0:HEIGHT:2, 0:WIDTH:2].astype(np.float32)
COLORS = {1: EMBER, -1: FROST}


def stroke(layer, points, color, width=2, closed=False):
    if len(points) >= 2:
        vertices = np.rint(points).astype(np.int32)
        cv2.polylines(layer, [vertices], closed, color, width, cv2.LINE_AA)


def draw_seal(layer, seal, color, now, strength=1, scale=1):
    center = point(*seal.center)
    radius = seal.radius * scale
    angle = now * (seal.direction or 1)
    color = tint(color, strength)
    white = tint(WHITE, strength)
    for points in (seal.outline, seal.glyph):
        scaled = [(center[0] + (x - center[0]) * scale, center[1] + (y - center[1]) * scale)
                  for x, y in points]
        stroke(layer, scaled, tint(color, 0.72), 6, points is seal.outline)
        stroke(layer, scaled, white, 2, points is seal.outline)
        head = int(now * 55) % len(scaled)
        ring(layer, point(*scaled[head]), 4, white, -1)
    ring(layer, center, radius + 10, tint(color, 0.42))
    for i in range(40):
        phase = i * tau / 40 - angle * 0.53
        line(layer, polar(center, radius + 12, phase),
             polar(center, radius + (21 if i % 5 == 0 else 16), phase), tint(color, 0.74))
    for i in range(6):
        phase = i * tau / 6 + angle * 0.47
        theta = phase * 180 / pi
        arc(layer, center, radius + 28, theta + 7, theta + 49, color, 2)
        sides = [polar(center, radius + d, phase + a) for d, a in ((34, 0), (28, .035), (22, 0), (28, -.035))]
        polygon(layer, sides, white, i % 2 == 0)


class Vortex:
    impact = 1.23
    duration = 3.35
    start_sound = "vortex"
    impact_sound = "rupture"
    title = "VORTEX"

    def __init__(self, seal, now):
        self.seal = seal
        self.started = now
        self.impacted = False
        self.color = COLORS[seal.direction]
        grid_y, grid_x = HALF_GRID
        x, y = seal.center
        self.dx, self.dy = grid_x - x, grid_y - y
        self.distance = np.hypot(self.dx, self.dy)
        reach = seal.radius * 3.5 + 70
        self.field = np.maximum(0, 1 - self.distance / reach) ** 2
        rng = np.random.default_rng(741)
        self.sparks = rng.uniform(size=(240, 4))

    def warp(self, frame, age):
        sign = self.seal.direction
        before = min(1, age / self.impact)
        after = max(0, age - self.impact)
        decay = exp(-after * 5)
        twist = self.field * (-sign * 3.9 * before ** 2 * decay)
        shock_radius = 28 + after * 845
        shock = np.exp(-((self.distance - shock_radius) / 31) ** 2)
        recoil = -sign * 0.18 * shock * decay if after > 0 else 0
        twist += recoil
        zoom = 1 + self.field * (0.44 * before * decay)
        warped = self.remap(frame, twist, zoom)
        flash = max(0, 1 - after / 0.18) if after > 0 else 0
        if flash:
            warped = cv2.addWeighted(warped, 1, np.full_like(warped, self.color), flash * 0.20, 0)
        return warped

    def remap(self, frame, twist, zoom):
        center_x, center_y = self.seal.center
        cos_turn, sin_turn = np.cos(twist), np.sin(twist)
        grid_x = (self.dx * cos_turn - self.dy * sin_turn) * zoom + center_x
        grid_y = (self.dx * sin_turn + self.dy * cos_turn) * zoom + center_y
        maps_x = cv2.resize(grid_x.astype(np.float32, copy=False), (WIDTH, HEIGHT))
        maps_y = cv2.resize(grid_y.astype(np.float32, copy=False), (WIDTH, HEIGHT))
        return cv2.remap(frame, maps_x, maps_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def draw(self, layer, age):
        if age < self.impact:
            self.draw_pull(layer, age)
        else:
            self.draw_shock(layer, age - self.impact)

    def draw_pull(self, layer, age):
        seal = self.seal
        center = point(*seal.center)
        time = age / self.impact
        draw_seal(layer, seal, self.color, age * 2, 1 - 0.26 * time, 1 + 0.20 * sin(pi * time))
        for angle, reach, delay, size in self.sparks:
            phase = max(0, min(1, (time - delay * 0.48) / (1 - delay * 0.48)))
            if phase <= 0:
                continue
            theta = angle * tau + seal.direction * (phase * 5.5 + 0.35 * sin(phase * tau))
            radius = 8 + (100 + reach * 445) * (1 - phase) ** 1.32
            tail = polar(center, radius + 7 + phase * 43, theta - seal.direction * (0.032 + phase * 0.14))
            head = polar(center, radius, theta)
            color = tint(self.color, 0.28 + size * 0.69)
            line(layer, tail, head, color, 1 + int(size > .8))
            if size > .84:
                ring(layer, head, 2, WHITE, -1)
        for i in range(4):
            angle = (age * 175 * seal.direction + i * 90) % 360
            size = seal.radius * (1.77 + 0.3 * sin(age * 5 + i))
            arc(layer, center, size, angle, angle + 33, tint(self.color, 0.58), 2)
        ring(layer, center, 4 + time ** 3 * 21, self.color, -1)
        ring(layer, center, 2 + time ** 3 * 10, WHITE, -1)

    def draw_shock(self, layer, age):
        center = point(*self.seal.center)
        intensity = exp(-age * 3.2)
        for i, speed in enumerate((820, 633, 440, 245)):
            radius = 15 + age * speed + 10 * i
            if radius < hypot(WIDTH, HEIGHT) and intensity > 0.02:
                ring(layer, center, radius, tint(self.color, intensity * (1 - i * 0.14)), 2)
                theta = i * 72 + age * 105 * self.seal.direction
                arc(layer, center, radius + 8, theta - 55, theta + 90, tint(WHITE, intensity), 2)
        for angle, reach, delay, size in self.sparks:
            theta = angle * tau + self.seal.direction * age * (0.18 + delay * 0.32)
            radius = (45 + reach * 500) * (1 - exp(-age * 3.2))
            p = polar(center, radius, theta)
            color = tint(self.color if size < 0.84 else WHITE, intensity * (0.55 + size * 0.45))
            if size > 0.93:
                polygon(layer, [polar(p, 2 + 7 * intensity, theta + i * tau / 3) for i in range(3)], color)
            tail = polar(center, max(0, radius - 22 - 34 * intensity * size), theta - self.seal.direction * 0.04)
            line(layer, tail, p, color, 1 + int(size > .86))
        if age < 0.20:
            ring(layer, center, 16 + age * 220, tint(WHITE, 1 - age / 0.20), -1)


class PalmVortex(Vortex):
    duration = 6.0
    fade_from = 3.4
    impact_sound = None
    title = "PALM VORTEX"

    def rotation(self, age):
        return self.seal.direction * age * 4.6

    def opacity(self, age):
        start = min(1.0, max(0.0, age / 0.58))
        fade = min(1.0, max(0.0, (age - self.fade_from) / (self.duration - self.fade_from)))
        return start * start * (3 - 2 * start) * (1 - fade * fade * (3 - 2 * fade))

    def warp(self, frame, age):
        opacity = self.opacity(age)
        if opacity <= 0:
            return frame
        # Continue advancing the sampling angle throughout the dissolve.
        twist = -self.field * self.rotation(age)
        warped = self.remap(frame, twist, 1 + self.field * 0.25)
        return cv2.addWeighted(frame, 1 - opacity, warped, opacity, 0)

    def draw(self, layer, age):
        opacity = self.opacity(age)
        if opacity <= 0:
            return
        seal = self.seal
        center = point(*seal.center)
        angle = self.rotation(age)
        draw_seal(layer, seal, self.color, age * 1.7, opacity * 0.72)
        for petal in range(5):
            root = petal * tau / 5 + angle
            trail = [polar(center, seal.radius * (1.88 - t * 1.72),
                           root + seal.direction * t * 1.86)
                     for t in np.linspace(0, 1, 28)]
            stroke(layer, trail, tint(self.color, 0.78 * opacity), 4)
            stroke(layer, trail, tint(WHITE, opacity), 1)
            ring(layer, trail[0], 5, tint(self.color, opacity), -1)
        for turn, reach, delay, size in self.sparks[:135]:
            cycle = (age * 0.62 + delay) % 1
            theta = turn * tau + angle + seal.direction * cycle * 2.1
            radius = 10 + (82 + reach * 330) * (1 - cycle) ** 1.32
            intensity = opacity * sin(pi * cycle) * (0.28 + size * 0.68)
            head = polar(center, radius, theta)
            tail = polar(center, radius + 10 + size * 26, theta - seal.direction * 0.13)
            line(layer, tail, head, tint(WHITE if size > 0.86 else self.color, intensity), 1)
        ring(layer, center, 7, tint(WHITE, opacity), -1)


class Lightning:
    impact = 0.66
    duration = 2.33
    start_sound = "lightning_charge"
    impact_sound = "thunder"
    title = "LIGHTNING"

    def __init__(self, seal, now):
        self.seal = seal
        self.started = now
        self.impacted = False
        nodes = np.asarray(seal.glyph)
        length = np.r_[0, np.linalg.norm(np.diff(nodes, axis=0), axis=1).cumsum()]
        steps = np.linspace(0, length[-1], 74)
        self.path = np.column_stack([np.interp(steps, length, nodes[:, axis]) for axis in (0, 1)])

    def warp(self, frame, age):
        pulse = max(0, 1 - abs(age - self.impact - 0.05) / 0.13)
        if pulse <= 0:
            return frame
        shift = round(pulse * 10)
        blue, green, red = cv2.split(frame)
        shifted = cv2.merge((np.roll(blue, shift, axis=1), green, np.roll(red, -shift, axis=1)))
        return cv2.addWeighted(shifted, 1.0, np.full_like(frame, VIOLET), pulse * 0.27, 0)

    def draw(self, layer, age):
        center = point(*self.seal.center)
        charge = min(1, age / self.impact)
        after = max(0, age - self.impact)
        fade = exp(-after * 2.7)
        rng = np.random.default_rng(int(age * 23) + 271)
        draw_seal(layer, self.seal, VIOLET, age * 2.7, (0.37 + 0.63 * charge) * fade)
        count = max(2, round(charge * len(self.path)))
        bolt = self.path[:count] + rng.uniform(-3, 3, (count, 2))
        stroke(layer, bolt, tint(VIOLET, fade), 8)
        stroke(layer, bolt, tint(WHITE, fade), 2)
        tip = point(*bolt[-1])
        ring(layer, tip, 5 + 10 * charge * fade, tint(VIOLET, fade), -1)
        ring(layer, tip, 3 + 4 * charge * fade, tint(WHITE, fade), -1)
        if age < self.impact:
            return
        for i in range(12):
            start = self.path[8 + i * 5]
            angle = i * tau / 12 + after * 0.32
            radius = (45 + (i * 47) % 140) * min(1, after * 8)
            branch = [tuple(start)]
            for j in range(1, 7):
                branch.append(polar(start, radius * j / 6, angle + rng.uniform(-0.2, 0.2)))
            stroke(layer, branch, tint(VIOLET, fade * 0.9), 3)
            stroke(layer, branch, tint(WHITE, fade * 0.85))
        for i in range(3):
            radius = self.seal.radius + after * (520 - i * 88) + i * 11
            arc(layer, center, radius, i * 100, i * 100 + 253, tint(VIOLET, fade), 2)


class PinchPen:
    hold = 0.20
    close_gap = 0.23
    release_gap = 0.36

    def __init__(self):
        self.since = self.last_seen = None

    def update(self, now, hand):
        if self.last_seen is not None and now - self.last_seen > 0.15:
            self.since = self.last_seen = None
        if hand is None:
            return False
        threshold = self.close_gap if self.since is None else self.release_gap
        if hand.pinch is None or hand.pinch_gap > threshold:
            self.since = self.last_seen = None
            return False
        self.last_seen = now
        if self.since is None:
            self.since = now
        return now - self.since >= self.hold


class RuneScene:
    def __init__(self, pen=None):
        self.reset()
        if pen is not None:
            self.pen = pen

    def reset(self):
        self.rune = Rune()
        self.effect = None
        self.pen = PinchPen()

    def draw(self, now, frame, hand, preview, play):
        if self.effect and now - self.effect.started >= self.effect.duration:
            self.reset()
        tracked = (hand.pinch or hand.index) if hand else None
        stage_before = self.rune.stage
        inking = self.pen.update(now, hand)
        open_palm = bool(hand and hand.pose == "open" and self.pen.since is None)
        seal = self.rune.update(now, tracked, inking, open_palm)
        if seal:
            self.effect = Lightning(seal, now) if seal.kind == "lightning" else Vortex(seal, now)
            play(self.effect.start_sound)
        elif self.rune.stage != stage_before:
            play("ready" if self.rune.stage == "ready" else "summon")
        layer = np.zeros_like(frame)
        color = INK
        if self.rune.seal:
            color = VIOLET if self.rune.seal.kind == "lightning" else COLORS[self.rune.seal.direction]
        if self.effect:
            age = now - self.effect.started
            frame = self.effect.warp(frame, age)
            self.effect.draw(layer, age)
            if self.effect.impact_sound and age >= self.effect.impact and not self.effect.impacted:
                play(self.effect.impact_sound)
                self.effect.impacted = True
        else:
            self.draw_ink(layer, now, tracked, color)
        output = composite(frame, layer)
        self.draw_hud(output, now, color, preview)
        return output

    def draw_ink(self, layer, now, tracked, color):
        if self.rune.seal:
            draw_seal(layer, self.rune.seal, color, now)
            age = now - self.rune.ready_at
            if age < 0.34:
                ring(layer, point(*self.rune.center), self.rune.radius + age * 365,
                     tint(WHITE, 1 - age / 0.34), 2)
        else:
            for points in (self.rune.outline, self.rune.points):
                stroke(layer, points, tint(color, 0.62), 7, points is self.rune.outline)
                stroke(layer, points, WHITE, 2, points is self.rune.outline)
            if self.rune.paused and self.rune.points:
                anchor = point(*self.rune.points[-1])
                ring(layer, anchor, 10 + 3 * sin(now * 9), EMBER, 2)
                ring(layer, anchor, 4, WHITE, -1)
            if tracked is not None:
                pos = point(*tracked)
                drawing = self.rune.pen_down and not self.rune.paused
                ring(layer, pos, 7 if drawing else 5, color, 1)
                if drawing:
                    ring(layer, pos, 3, WHITE, -1)
                    for i in range(7):
                        theta = i * tau / 7 - now * 3
                        ring(layer, polar(pos, 9 + (now * 32 + i * 8) % 26, theta), 1, color, -1)

    def draw_hud(self, frame, now, color, preview):
        messages = {
            "ring": "Hold thumb and index together, then draw a circle",
            "glyph": "Pinch inside the circle and draw a hook or zigzag",
            "ready": "Open your hand to cast",
            "cast": self.effect.title if self.effect else "",
        }
        status = messages[self.rune.stage]
        if self.rune.ready_at >= 0 and self.rune.stage == "ready":
            status = ("Lightning ready" if self.rune.seal.kind == "lightning" else "Vortex ready") + "   /   " + status
        if self.rune.paused:
            status = "Pinch near the gold dot to keep drawing"
        for x, dx in ((22, 1), (937, -1)):
            for y, dy in ((22, 1), (518, -1)):
                line(frame, (x, y), (x + 19 * dx, y), color)
                line(frame, (x, y), (x, y + 19 * dy), color)
        cv2.rectangle(frame, (38, 35), (297, 101), (24, 15, 24), -1)
        line(frame, (38, 35), (38, 101), color, 2)
        label(frame, "RUNE SEALS", (52, 64), 0.80, WHITE, 2)
        label(frame, "EFFECTS PREVIEW" if preview else "LIVE CAMERA", (53, 86), .39, color)
        cv2.rectangle(frame, (38, 464), (921, 502), (24, 15, 24), -1)
        label(frame, status, (52, 488), 0.53)
        if self.rune.open_since is not None and self.rune.stage == "ready":
            charge = min(1, (now - self.rune.open_since) / .35)
            line(frame, (38, 501), (38 + round(690 * charge), 501), color, 2)
        label(frame, "Backspace Undo Stroke   R Reset   Esc Close", (610, 488), .38, (206, 183, 208))
        if self.rune.stage == "glyph":
            self.draw_choices(frame)

    def draw_choices(self, frame):
        cv2.rectangle(frame, (702, 35), (921, 115), (24, 15, 24), -1)
        spiral = [polar((752, 66), 17 - i * 0.55, -1.6 + i * 0.15) for i in range(25)]
        stroke(frame, spiral, EMBER, 2)
        stroke(frame, [(849, 56), (878, 56), (849, 76), (877, 76)], VIOLET, 2)
        label(frame, "Vortex", (727, 102), .41, WHITE)
        label(frame, "Lightning", (831, 102), .41, WHITE)


# Two-hand chakra and Overdrive

CYAN = (255, 226, 111)
CHAKRA_VIOLET = (255, 107, 186)
CHAKRA_GOLD = (105, 213, 255)
DARK = (23, 19, 15)
RIFT_DELAY = 0.43


class Chakra:
    charge_min = 1.72
    charge_max = 3.50

    def __init__(self):
        self.stage = "seal"
        self.power = 0.0
        self.hold_since = None
        self.previous = None
        self.angle = None
        self.turn = 0.0
        self.overdrive = False
        self.charging = False
        self.release_gap = 3.65

    @staticmethod
    def can_seal(gap):
        return 0.35 <= gap <= 1.76

    def update(self, now, gap, angle=None):
        previous, self.previous = self.previous, now
        dt = now - previous if previous is not None else 0
        if gap is None or dt > 0.18:
            self.hold_since = None
            self.angle = None
            self.charging = False
            return None
        if self.stage == "seal":
            if self.held(now, self.can_seal(gap), 0.22):
                self.stage = "charge"
                self.hold_since = None
                return "seal"
        elif self.stage == "charge":
            in_band = self.charge_min <= gap <= self.charge_max
            if self.charging and in_band:
                self.power = min(1, self.power + dt / 0.77)
            self.charging = in_band
            if self.power == 1:
                self.stage = "ready"
                self.angle = angle
                self.release_gap = min(3.85, max(3.15, gap + 0.68))
                return "ready"
        elif self.stage == "ready":
            release_gap = self.release_gap - (0.16 if self.hold_since is not None else 0)
            if self.held(now, gap >= release_gap, 0.14):
                self.stage = "release"
                self.hold_since = None
                return "release"
            if not self.overdrive and self.twist(gap, angle):
                self.overdrive = True
                return "overdrive"
        return None

    def twist(self, gap, angle):
        previous = self.angle
        self.angle = angle if 1.68 <= gap < self.release_gap else None
        if previous is None or self.angle is None:
            return False
        # A pair of palms defines an axis, so swapping its ends is a half-turn, not motion.
        delta = 2 * (self.angle - previous)
        step = atan2(sin(delta), cos(delta)) / 2
        if abs(step) <= pi / 6:
            self.turn += step
        return abs(self.turn) >= pi / 4

    def held(self, now, condition, duration):
        if not condition:
            self.hold_since = None
            return False
        if self.hold_since is None:
            self.hold_since = now
        return now - self.hold_since >= duration


def palm_pair(hands):
    if len(hands) != 2:
        return None
    (left, left_scale), (right, right_scale) = sorted(hands, key=lambda hand: hand[0][0])
    dx, dy = right[0] - left[0], right[1] - left[1]
    scale = (left_scale + right_scale) / 2
    if scale < 18 or min(left_scale, right_scale) < 0.42 * max(left_scale, right_scale):
        return None
    center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    return center, hypot(dx, dy) / scale, scale, atan2(dy, dx)


class ChakraScene:
    flight = 0.34

    def __init__(self):
        self.sparks = np.random.default_rng(1024).uniform(size=(140, 3))
        self.reset()

    def reset(self):
        self.chakra = Chakra()
        self.center = None
        self.release_time = None
        self.radius = 40.0
        self.angle = 0.0
        self.ruptured = False
        self.aim_origin = None
        self.shot_target = None

    def draw(self, now, frame, hands, preview, play):
        if self.finished(now):
            self.reset()
        pair = palm_pair(hands)
        event = self.chakra.update(now, pair[1] if pair else None, pair[3] if pair else None)
        if pair and self.release_time is None:
            center, gap, scale, self.angle = pair
            self.center = (tuple(previous * 0.54 + current * 0.46
                                 for previous, current in zip(self.center, center))
                           if self.center else center)
            self.radius = max(30, min(69, scale * 0.75))
            if self.chakra.stage == "ready" and gap < self.chakra.release_gap - 0.16:
                self.aim_origin = self.center
        if event == "seal":
            play("summon")
        elif event == "ready":
            play("chakra_ready")
        elif event == "overdrive":
            play("chakra_overdrive")
        elif event == "release":
            self.release_time = now
            self.shot_target = self.aim_target()
            play("chakra_collapse" if self.chakra.overdrive else "chakra_release")

        layer = np.zeros_like(frame)
        if self.release_time is not None:
            age = now - self.release_time
            impact_age = age - (RIFT_DELAY if self.chakra.overdrive else self.flight)
            if self.chakra.overdrive:
                frame = self.lens(frame, self.rift_strength(age))
                self.draw_rift(layer, age)
                if impact_age >= 0 and not self.ruptured:
                    play("chakra_rift")
                    self.ruptured = True
            elif impact_age < 0:
                self.draw_shot(layer, age / self.flight)
            else:
                self.draw_release(layer, impact_age, self.shot_target)
            if 0 <= impact_age < 0.20:
                flash = (1 - impact_age / 0.20) ** 2
                frame = cv2.addWeighted(frame, 1, np.full_like(frame, CYAN), flash * 0.37, 0)
        elif self.center is not None and self.chakra.stage != "seal":
            if self.chakra.overdrive:
                frame = self.lens(frame, 0.25 + 0.04 * sin(now * 7))
                self.draw_twins(layer, now)
            else:
                self.draw_core(layer, now)
        if self.chakra.stage != "release":
            for palm, scale in hands:
                center = point(*palm)
                ring(layer, center, max(7, scale * 0.18), tint(CYAN, 0.55))
                ring(layer, center, 2, CYAN, -1)
        output = composite(frame, layer)
        self.draw_hud(output, pair, preview)
        return output

    def finished(self, now):
        duration = 2.80 if self.chakra.overdrive else 2.30
        return self.release_time is not None and now - self.release_time > duration

    def aim_target(self):
        origin = self.aim_origin if self.aim_origin is not None else self.center
        dx, dy = self.center[0] - origin[0], self.center[1] - origin[1]
        angle = atan2(dy, dx) if hypot(dx, dy) >= max(12, self.radius * 0.22) else self.angle
        x, y = polar(self.center, 390, angle)
        return point(min(WIDTH - 40, max(40, x)), min(HEIGHT - 52, max(52, y)))

    def draw_shot(self, layer, progress):
        dx = self.shot_target[0] - self.center[0]
        dy = self.shot_target[1] - self.center[1]
        angle = atan2(dy, dx)
        progress = progress ** 1.22
        center = point(self.center[0] + dx * progress, self.center[1] + dy * progress)
        length = 34 + 162 * progress
        radius = self.radius * (0.47 - 0.19 * progress)
        tail = polar(center, length, angle + pi)
        line(layer, tail, center, CYAN, 5)
        line(layer, polar(center, length * 0.70, angle + pi), center, WHITE, 2)
        for i in range(5):
            phase = 22 * progress + i * tau / 5
            p = polar(center, radius * 1.12, phase)
            q = polar(p, length * (0.22 + i * 0.12), angle + pi + sin(phase) * 0.25)
            line(layer, q, p, (CYAN, CHAKRA_GOLD, CHAKRA_VIOLET)[i % 3], 2)
        ring(layer, center, radius, CYAN, -1)
        ring(layer, center, radius * 0.52, WHITE, -1)

    def lens(self, frame, strength):
        x, y = map(np.float32, self.center)
        dx, dy = HALF_GRID[1] - x, HALF_GRID[0] - y
        reach = np.hypot(dx, dy) / (160 + self.radius * 1.4)
        field = np.exp(-reach ** 2 * 2)
        twist = field * strength * (1 if self.chakra.turn >= 0 else -1)
        shrink = 1 + field * abs(strength) * 0.30
        sine, cosine = np.sin(twist), np.cos(twist)
        map_x = (x + (dx * cosine - dy * sine) * shrink).astype(np.float32, copy=False)
        map_y = (y + (dx * sine + dy * cosine) * shrink).astype(np.float32, copy=False)
        maps_x = cv2.resize(map_x, (WIDTH, HEIGHT))
        maps_y = cv2.resize(map_y, (WIDTH, HEIGHT))
        return cv2.remap(frame, maps_x, maps_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def rift_strength(self, age):
        if age < RIFT_DELAY:
            return 0.25 + 2.6 * (age / RIFT_DELAY) ** 2
        return -1.12 * exp(-7 * (age - RIFT_DELAY))

    def draw_twins(self, layer, now):
        center = point(*self.center)
        spin = ((now * 4.5) + self.angle) % tau
        for i, color in enumerate((CYAN, CHAKRA_GOLD)):
            phase = spin + i * pi
            core = polar(center, self.radius * 0.72, phase)
            arc(layer, center, self.radius * 0.72, phase * 180 / pi - 136, phase * 180 / pi,
                tint(color, 0.70), 3)
            ring(layer, core, self.radius * 0.45, tint(color, 0.38), -1)
            for strand in range(7):
                angle = phase + strand * 0.72
                start = (angle * 130) % 360
                axes = (round(self.radius * 0.58), round(self.radius * (0.13 + 0.045 * strand)))
                cv2.ellipse(layer, core, axes, angle * 180 / pi, start, start + 156, color,
                            1 + (strand % 3 == 0), cv2.LINE_AA)
            ring(layer, core, 5, WHITE, -1)
        for i in range(5):
            angle = (spin * (-1 if i % 2 else 1) * 180 / pi + i * 75) % 360
            arc(layer, center, self.radius * (1.12 + i * 0.16), angle, angle + 100,
                tint((CYAN, CHAKRA_VIOLET, CHAKRA_GOLD)[i % 3], 0.72), 2)

    def draw_rift(self, layer, age):
        center = point(*self.center)
        time = age - RIFT_DELAY
        if time < 0:
            fraction = age / RIFT_DELAY
            for turn, reach, strength in self.sparks:
                angle = turn * tau + fraction * 4
                radius = 6 + (85 + reach * 520) * (1 - fraction) ** 1.7
                head = polar(center, radius, angle)
                tail = polar(center, radius + 14 + strength * 38, angle - 0.13)
                line(layer, tail, head, tint(CYAN if strength < 0.6 else CHAKRA_GOLD, 0.5 + strength * 0.5), 2)
            ring(layer, center, max(3, 43 * (1 - fraction)), WHITE, 2)
            return
        self.draw_release(layer, time)
        fade = exp(-time * 3.1)
        for i, color in enumerate((CYAN, CHAKRA_GOLD, CHAKRA_VIOLET)):
            extent = min(660, (i * 36 + 940 * time))
            phase = self.angle + i * pi / 3 + time * 0.68
            for direction in (0, pi):
                angle = phase + direction
                streak = [polar(center, extent * s / 15, angle + sin(s / 15 * pi) * 0.21)
                          for s in range(16)]
                path = np.asarray(streak, dtype=np.int32)
                cv2.polylines(layer, [path], False, tint(color, fade), 6 - i, cv2.LINE_AA)
                cv2.polylines(layer, [path], False, tint(WHITE, fade), 1, cv2.LINE_AA)

    def draw_core(self, layer, now):
        center = point(*self.center)
        power = self.chakra.power
        ready = self.chakra.stage == "ready"
        radius = self.radius * (0.54 + 0.63 * power) * (1 + 0.045 * sin(now * 13))
        speed = now * (4.2 + power * 5.0)
        glow = CHAKRA_GOLD if ready else CYAN
        ring(layer, center, radius * 0.81, (25, 65, 75), -1)
        ring(layer, center, radius + 2, tint(glow, 0.56), 2)
        for i in range(13):
            phase = (speed * (1 if i % 2 else -1) + i * 1.2) % tau
            height = radius * (0.19 + 0.7 * abs(sin(phase + i)))
            axes = (round(radius * (0.96 + i * 0.020)), max(2, round(height)))
            angle = (self.angle + i * pi / 7 + speed * 0.36) * 180 / pi
            start = (phase * 180 / pi) % 360
            length = 79 + i * 11
            color = (CYAN, CHAKRA_VIOLET, glow)[i % 3]
            cv2.ellipse(layer, center, axes, angle, start, start + length,
                        tint(color, 0.47 + power * 0.38), 1 + (i % 3 == 0), cv2.LINE_AA)
            if i % 3 == 0:
                cv2.ellipse(layer, center, axes, angle, start + 12, start + length * 0.55,
                            WHITE, 1, cv2.LINE_AA)
        for i in range(4):
            phase = speed * (-0.48 if i % 2 else 0.4) + i * pi / 2
            tip = polar(center, radius + 15 + 14 * power, phase)
            arc(layer, center, radius + 15 + 14 * power, phase * 180 / pi - 38,
                phase * 180 / pi, tint(glow, 0.6))
            ring(layer, tip, 2 + power * 2, WHITE, -1)
        for base, reach, strength in self.sparks[:int(35 + 65 * power)]:
            orbit = ((now * (0.56 + strength * 0.5) + reach) % 1)
            angle = base * tau + orbit * tau * 0.75
            distance = radius + 110 * (1 - orbit) ** 2
            spark = polar(center, distance, angle)
            line(layer, polar(center, distance + 6 + 12 * strength, angle - 0.09), spark,
                 tint(glow if strength > 0.55 else CHAKRA_VIOLET, 0.25 + 0.65 * strength))
        ring(layer, center, 3 + 6 * power, WHITE, -1)

    def draw_release(self, layer, age, target=None):
        center = target if target is not None else point(*self.center)
        fade = exp(-age * 2.45)
        for i, speed in enumerate((1020, 785, 565, 370)):
            radius = 22 + age * speed
            if radius < hypot(WIDTH, HEIGHT):
                color = (CYAN, WHITE, CHAKRA_VIOLET, CHAKRA_GOLD)[i]
                ring(layer, center, radius, tint(color, fade * (1 - i * 0.12)), 2)
                angle = age * 125 + i * 53
                arc(layer, center, radius + 9, angle, angle + 208, tint(color, fade * 0.70), 2)
        for turn, reach, strength in self.sparks:
            angle = turn * tau + age * 0.23
            distance = 16 + (130 + 670 * reach) * (1 - exp(-3.2 * age))
            head = polar(center, distance, angle)
            tail = polar(center, max(0, distance - 14 - 62 * strength * fade), angle - 0.03)
            line(layer, tail, head, tint(CYAN if strength < .80 else WHITE, fade * (0.3 + strength)), 2)
        if age < 0.20:
            ring(layer, center, 17 + 230 * age, tint(WHITE, 1 - age / 0.20), -1)

    def draw_hud(self, frame, pair, preview):
        stage = self.chakra.stage
        status = {
            "seal": "Hold your hands close with a small gap",
            "charge": "Separate your hands a little and hold",
            "ready": ("OVERDRIVE / spread your hands wide" if self.chakra.overdrive
                      else "Tilt both hands to power up, or spread to release"),
            "release": "RIFT BURST" if self.chakra.overdrive else "CHAKRA SHOT",
        }[stage]
        if stage == "charge" and pair:
            gap = pair[1]
            if gap < self.chakra.charge_min:
                status = "Move your hands slightly farther apart"
            elif gap > self.chakra.charge_max:
                status = "Bring your hands a little closer to charge"
        if stage != "release" and pair is None:
            status = "Show both hands" if stage == "seal" else "Bring both hands back into view"
        cv2.rectangle(frame, (34, 32), (282, 100), DARK, -1)
        line(frame, (34, 32), (34, 100), CYAN, 2)
        label(frame, "CHAKRA SURGE", (48, 61), 0.77, WHITE, 2)
        label(frame, "GESTURE PREVIEW" if preview else "LIVE CAMERA", (49, 85), 0.40, CYAN)
        cv2.rectangle(frame, (34, 462), (925, 504), DARK, -1)
        label(frame, status, (48, 489), 0.51)
        if stage in ("charge", "ready"):
            color = CHAKRA_GOLD if stage == "ready" else CYAN
            line(frame, (520, 483), (708, 483), (75, 74, 60), 3)
            if self.chakra.power:
                line(frame, (520, 483), (520 + round(188 * self.chakra.power), 483), color, 3)
            meter = "MAX" if self.chakra.overdrive else f"{self.chakra.power:.0%}"
            label(frame, meter, (721, 489), 0.46, color)
        label(frame, "R Reset   Esc Close", (791, 489), 0.42, (197, 192, 181))


def demo_palms(now):
    phase = now % 6.6
    if phase < 0.20 or phase > 4.55:
        return ()
    gap = np.interp(phase, (0.2, 0.65, 1.22, 3.04, 3.41, 4.55), (1.15, 1.15, 2.64, 2.64, 5.0, 5.0))
    angle = np.interp(phase, (0.2, 2.15, 2.90, 4.55), (0.04, 0.04, 1.09, 1.09))
    size = 73
    center_x, center_y = 480, 267
    x = cos(angle) * gap * size / 2
    y = sin(angle) * gap * size / 2
    return (((center_x - x, center_y - y), size), ((center_x + x, center_y + y), size))


# Open-palm vortex and teleportation

class PalmTwist:
    required = radians(42)

    def __init__(self):
        self.reset()

    def reset(self):
        self.angle = None
        self.turn = 0.0
        self.last_seen = None

    def update(self, now, hand):
        if hand is None or hand.pose != "open":
            self.angle = None
            if self.last_seen is not None and now - self.last_seen > 0.18:
                self.reset()
            return None
        if self.last_seen is not None and now - self.last_seen > 0.18:
            self.reset()
        self.last_seen = now
        if self.angle is not None:
            delta = hand.angle - self.angle
            step = atan2(sin(delta), cos(delta))
            if abs(step) <= radians(38):
                self.turn += step
        self.angle = hand.angle
        if abs(self.turn) < self.required:
            return None
        direction = 1 if self.turn > 0 else -1
        radius = max(53, min(112, hand.palm_size * 1.16))
        outline = [polar(hand.palm, radius, phase) for phase in np.linspace(0, tau, 69)]
        glyph = [polar(hand.palm, radius * (0.90 - t * 0.83),
                       hand.angle + direction * t * tau * 1.42)
                 for t in np.linspace(0, 1, 90)]
        return Seal(hand.palm, radius, outline, glyph, "vortex", direction)

    def draw_hint(self, frame, hand):
        center = point(*hand.palm)
        charge = min(1, abs(self.turn) / self.required)
        radius = max(29, min(78, hand.palm_size * 0.82))
        color = EMBER if self.turn >= 0 else FROST
        layer = np.zeros_like(frame)
        for i in range(5):
            angle = hand.angle + i * tau / 5
            degrees = angle * 180 / pi
            arc(layer, center, radius + 12, degrees, degrees + 39, color, 2)
            tip = polar(center, radius * (0.94 + charge * 0.1), angle)
            ring(layer, tip, 4 + charge * 2, color)
        arc(layer, center, radius - 3, -90, -90 + charge * 360, WHITE, 2)
        return composite(frame, layer)


def near_forehead(hand, face):
    if hand.pose != "sign" or face is None or face.width < 44:
        return False
    index_x, index_y = hand.index
    middle_x = 2 * hand.fingers[0] - index_x
    middle_y = 2 * hand.fingers[1] - index_y
    if hypot(index_x - middle_x, index_y - middle_y) > face.width * 0.37:
        return False
    return all(face.x - face.width * 0.08 <= x <= face.x + face.width * 1.08
               and face.y - face.height * 0.21 <= y <= face.y + face.height * 0.40
               for x, y in ((index_x, index_y), (middle_x, middle_y)))


class TeleportScene:
    duration = 2.76
    reappear_at = 2.36
    blink = 0.16

    def __init__(self, now, frame, background, face, anchor):
        self.started = now
        self.departure = frame.copy()
        self.background = background
        self.anchor = point(*anchor)
        self.return_sound = False
        self.x0 = max(0, int(face.x - face.width * 1.32))
        self.x1 = min(WIDTH, int(face.x + face.width * 2.34))
        self.y0 = max(0, int(face.y - face.height * 0.38))
        self.y1 = min(HEIGHT, int(face.y + face.height * 4.1))
        rng = np.random.default_rng(3189)
        self.shreds = rng.uniform(size=(72, 4))
        self.rows = rng.uniform(0.08, 0.77, (HEIGHT + 7) // 8).repeat(8)[:HEIGHT]

    def draw(self, now, frame, play):
        age = now - self.started
        if age < self.blink:
            return self.transition(self.departure, age / self.blink, False)
        if age < self.reappear_at:
            return self.background
        if not self.return_sound:
            play("teleport")
            self.return_sound = True
        progress = (age - self.reappear_at) / self.blink
        return self.transition(frame, 1 - progress, True) if progress < 1 else frame

    def transition(self, person, wipe, returning):
        blend_wide = wipe * wipe * (3 - 2 * wipe)
        base = cv2.addWeighted(person, 1 - blend_wide, self.background, blend_wide, 0)
        partial = np.clip((wipe - self.rows) * 7.5, 0, 1)[:, None, None]
        rows, cols = slice(self.y0, self.y1), slice(self.x0, self.x1)
        old = person[rows, cols]
        clean = self.background[rows, cols]
        blend = partial[rows]
        base[rows, cols] = (old * (1 - blend) + clean * blend).astype(np.uint8)
        layer = np.zeros_like(base)
        motion = 1 - wipe if returning else wipe
        ease = sin(pi * motion) ** 0.65
        accent = (253, 241, 221)
        for height, span, shift, strength in self.shreds:
            y = self.y0 + round(height * (self.y1 - self.y0))
            x = self.x0 + round(shift * (self.x1 - self.x0))
            reach = (28 + span * 176) * (0.18 + motion)
            jog = (-1 if returning else 1) * motion * (26 + 142 * span)
            color = tint(WHITE if strength > 0.62 else accent, ease * (0.45 + 0.55 * strength))
            if strength < 0.29:
                shadow = point(x + jog - 32, y + 1)
                line(base, shadow, point(x + jog + reach, y + 1), (16, 14, 18), 1)
            line(layer, point(x + jog, y), point(x + jog + reach, y), color,
                 1 + int(strength > 0.83))
        x, y = self.anchor
        flash = max(0, 1 - abs(motion - 0.24) / 0.34)
        line(layer, point(x - 98 * flash, y), point(x + 98 * flash, y), tint(accent, flash), 2)
        if flash > 0.1:
            ring(layer, self.anchor, 4 + flash * 12, tint(WHITE, flash), -1)
        return composite(base, layer)


# Blue and Red fusion

PURPLE_BLUE = (255, 136, 31)
PURPLE_RED = (46, 57, 255)
PURPLE_LIGHT = (255, 102, 226)
PURPLE_WHITE = (255, 230, 255)


def purple_pair(hands):
    if len(hands) != 2:
        return None
    return palm_pair([(hand.palm, hand.palm_size) for hand in hands])


def purple_sign(hands):
    pair = purple_pair(hands)
    if pair is None or pair[1] < 2.05:
        return False
    left, right = sorted(hands, key=lambda hand: hand.palm[0])
    return (left.pose == "open" or left.open_fingers >= 3) and right.pose == "point"


class PurpleCharge:
    def __init__(self):
        self.blue_at = self.red_at = None
        self.close_since = None
        self.fused = False
        self.last_pair_at = None

    def update(self, now, hands):
        if self.fused:
            return None
        pair = purple_pair(hands)
        if pair is None:
            self.close_since = self.last_pair_at = None
            return None
        if self.last_pair_at is not None and now - self.last_pair_at > 0.18:
            self.close_since = None
        self.last_pair_at = now
        left, right = sorted(hands, key=lambda hand: hand.palm[0])
        if self.blue_at is None:
            if left.pose == "open" or left.open_fingers >= 3:
                self.blue_at = now
                return "blue"
            return None
        if self.red_at is None:
            if now - self.blue_at >= 0.34 and right.pose == "point":
                self.red_at = now
                return "red"
            return None
        if pair[1] > 1.90:
            self.close_since = None
        elif self.close_since is None:
            self.close_since = now
        elif now - self.close_since >= 0.13:
            self.fused = True
            return "fuse"
        return None


class PurpleScene:
    duration = 2.80
    merge_time = 0.57
    flight = 0.42

    def __init__(self, started=False):
        self.sparks = np.random.default_rng(8419).uniform(size=(235, 4))
        self.reset()
        self.started = started

    def reset(self):
        self.charge = PurpleCharge()
        self.started = False
        self.sign_since = None
        self.anchors = None
        self.fusion_at = None
        self.origin = (WIDTH / 2, HEIGHT / 2)
        self.size = 57
        self.hold_position = None
        self.right_side = None
        self.right_seen = None
        self.release_armed = False
        self.release_since = None
        self.ready_played = False
        self.shot_at = None
        self.impact_played = False

    def finished(self, now):
        return self.shot_at is not None and now - self.shot_at >= self.duration

    def draw(self, now, frame, hands, preview, play):
        if self.finished(now):
            self.reset()
        if not self.started:
            if not purple_sign(hands):
                self.sign_since = None
            elif self.sign_since is None:
                self.sign_since = now
            elif now - self.sign_since >= 0.20:
                self.started = True
        pair = purple_pair(hands)
        layer = np.zeros_like(frame)
        if self.started and self.fusion_at is None:
            if pair:
                left, right = sorted(hands, key=lambda hand: hand.palm[0])
                points = ((left.palm[0], left.palm[1] - pair[2] * 0.76),
                          (right.index[0], right.index[1]))
                if self.anchors:
                    points = tuple(tuple(0.51 * old + 0.49 * new for old, new in zip(a, b))
                                   for a, b in zip(self.anchors, points))
                self.anchors = points
                self.size = max(37, min(78, pair[2] * 0.92))
            event = self.charge.update(now, hands)
            if event == "blue":
                play("purple_form")
            elif event == "red":
                play("purple_red")
            elif event == "fuse":
                self.fusion_at = now
                self.origin = tuple((a + b) / 2 for a, b in zip(*self.anchors))
                self.right_side = right.handedness
                self.hold_position = self.origin
                play("purple_fuse")
        if self.shot_at is not None:
            age = now - self.shot_at
            if age >= self.flight and not self.impact_played:
                play("purple_burst")
                self.impact_played = True
            self.draw_shot(layer, age, now)
        elif self.fusion_at is not None:
            self.follow_hand(now, hands, play)
            if self.shot_at is None:
                age = now - self.fusion_at
                if age < self.merge_time:
                    self.draw_fusion(layer, age, now)
                else:
                    self.draw_held(layer, self.hold_position, now, min(1, (age - self.merge_time) / 0.44))
            else:
                self.draw_shot(layer, 0, now)
        elif self.started and self.anchors:
            for center, color, direction, formed in zip(
                    self.anchors, (PURPLE_BLUE, PURPLE_RED), (1, -1), (self.charge.blue_at, self.charge.red_at)):
                if formed is not None:
                    bloom = min(1, (now - formed) / (0.43 if direction > 0 else 0.27))
                    self.draw_orb(layer, center, self.size * (0.11 + 0.84 * bloom), color,
                                  direction, now, 0.37 + 0.63 * bloom)
            if self.charge.red_at is not None:
                self.draw_bridge(layer, now)
        output = composite(frame, layer)
        self.draw_hud(output, pair, preview)
        return output

    def follow_hand(self, now, hands, play):
        if self.right_side is not None:
            right = next((hand for hand in hands if hand.handedness == self.right_side), None)
        else:
            right = max(hands, key=lambda hand: hand.palm[0]) if len(hands) == 2 else None
        if right is not None:
            if self.right_seen is not None and now - self.right_seen > 0.18:
                self.release_since = None
            goal = (right.palm[0], right.palm[1] - self.size * 1.11)
            self.hold_position = tuple(0.59 * a + 0.41 * b for a, b in zip(self.hold_position, goal))
            self.right_seen = now
        if now - self.fusion_at < self.merge_time:
            return
        if not self.ready_played:
            self.ready_played = True
            play("purple_ready")
        if right is None:
            self.release_since = None
            if self.right_seen is not None and now - self.right_seen > 0.24:
                self.release_armed = False
            return
        if right.pose == "point":
            self.release_armed = True
        if not self.release_armed or right.pose != "open":
            self.release_since = None
        elif self.release_since is None:
            self.release_since = now
        elif now - self.release_since >= 0.13:
            self.shot_at = now

    def draw_orb(self, layer, location, radius, color, direction, now, strength):
        center = point(*location)
        ring(layer, center, radius * 0.69, tint(color, 0.29 * strength), -1)
        ring(layer, center, radius * 0.98, tint(color, strength), 3)
        for i in range(11):
            phase = now * (4.5 + i * 0.17) * direction + i * 1.14
            angle = (phase * 180 / pi) % 360
            axes = (round(radius * (0.79 + i * 0.047)), max(2, round(radius * (0.16 + 0.7 * abs(sin(phase))))))
            cv2.ellipse(layer, center, axes, angle, i * 31, i * 31 + 117 + i * 8,
                        tint(color, 0.8 * strength), 2, cv2.LINE_AA)
            if i % 3 == 0:
                cv2.ellipse(layer, center, axes, angle, i * 31 + 18, i * 31 + 76,
                            tint(PURPLE_WHITE, 0.84 * strength), 1, cv2.LINE_AA)
        for route, speed, delay, size in self.sparks[:84]:
            cycle = (now * (0.50 + speed * 0.5) + delay) % 1
            motion = 1 - cycle if direction > 0 else cycle
            travel = radius * (0.94 + (0.48 + size * 1.95) * motion)
            angle = route * tau + direction * (now * 2.5 + cycle * 2)
            intensity = sin(pi * cycle) * strength * (0.47 + size * 0.53)
            tail_offset = 8 + speed * 22
            tail = polar(center, travel + direction * tail_offset, angle - 0.13 * direction)
            line(layer, tail, polar(center, travel, angle), tint(color, intensity), 1 + int(size > 0.82))
        ring(layer, center, radius * 0.23, tint(color, strength), -1)
        ring(layer, center, radius * 0.11, tint(PURPLE_WHITE, strength), -1)

    def draw_bridge(self, layer, now):
        left, right = self.anchors
        for i, color in enumerate((PURPLE_BLUE, PURPLE_RED)):
            points = [(a + (b - a) * t for a, b in zip(left, right)) for t in np.linspace(0, 1, 22)]
            ribbon = [(x, y + sin(j * 0.67 + now * 19 + i * pi) * sin(j / 21 * pi) * 13)
                      for j, (x, y) in enumerate(points)]
            stroke(layer, ribbon, tint(color, 0.62), 2)

    def draw_fusion(self, layer, age, now):
        center = point(*self.origin)
        phase = age / self.merge_time
        for source, color, direction in zip(self.anchors, (PURPLE_BLUE, PURPLE_RED), (1, -1)):
            sit = tuple(a + (b - a) * phase ** 0.78 for a, b in zip(source, self.origin))
            self.draw_orb(layer, sit, self.size * (0.88 - phase * 0.55), color,
                          direction, now, 1 - 0.47 * phase)
        self.draw_infall(layer, center, phase)
        if phase > 0.72:
            t = (phase - 0.72) / 0.28
            self.draw_orb(layer, self.origin, self.size * (0.20 + 0.65 * t), PURPLE_LIGHT, 1, now, t)

    def draw_held(self, layer, location, now, growth, scale=1.0):
        center = point(*location)
        radius = self.size * (0.83 + 1.12 * growth) * scale * (1 + 0.029 * sin(now * 5))
        extent = round(radius * 1.48)
        x, y = center
        x0, x1 = max(0, x - extent), min(WIDTH, x + extent + 1)
        y0, y1 = max(0, y - extent), min(HEIGHT, y + extent + 1)
        if x0 < x1 and y0 < y1:
            yg = np.arange(y0 - y, y1 - y, 2, dtype=np.float32)[:, None]
            xg = np.arange(x0 - x, x1 - x, 2, dtype=np.float32)[None, :]
            distance = np.hypot(xg, yg) / radius
            angle = np.arctan2(yg, xg)
            spiral = (0.5 + 0.5 * np.cos(angle * 5 + distance * 31 + now * 7.8)) ** 5
            core = np.maximum(0, 1 - distance)
            aura = np.exp(-distance ** 2 * 3.7) * 0.29
            aura += spiral * core * 0.42 + np.exp(-((distance - 0.84) / 0.14) ** 2) * 0.13
            aura = np.clip(aura, 0, 1)
            light = (aura[:, :, None] * np.asarray(PURPLE_LIGHT)).astype(np.uint8)
            light = cv2.resize(light, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            base = layer[y0:y1, x0:x1]
            cv2.add(base, light, dst=base)
        for i in range(9):
            phase = now * (2.6 if i % 2 else -2.1) + i * tau / 9
            ring_radius = radius * (0.82 + i * 0.077)
            axes = (round(ring_radius), max(2, round(ring_radius * (0.21 + 0.48 * abs(sin(phase))))))
            angle = (phase * 180 / pi) % 360
            color = (PURPLE_LIGHT, (255, 176, 200), (242, 75, 190))[i % 3]
            start = i * 33
            cv2.ellipse(layer, center, axes, angle, start, start + 157, tint(color, 0.84), 3, cv2.LINE_AA)
            cv2.ellipse(layer, center, axes, angle, start + 19, start + 81,
                        tint(PURPLE_WHITE, 0.92), 1, cv2.LINE_AA)
        for i in range(10):
            offset = i * tau / 10 + now * 2.0
            bolt = [polar(center, radius * (0.68 + 0.064 * j + 0.032 * sin(j * 5 + i + now * 13)),
                          offset + 0.078 * j + 0.035 * sin(i + j * 7 + now * 21))
                    for j in range(8)]
            stroke(layer, bolt, tint(PURPLE_LIGHT, 0.79), 3)
            stroke(layer, bolt, tint(PURPLE_WHITE, 0.83))
        ring(layer, center, radius * 0.085, PURPLE_WHITE, -1)

    def draw_shot(self, layer, age, now):
        center = point(*self.hold_position)
        if age < self.flight:
            phase = age / self.flight
            scale = 1 + 1.35 * phase ** 1.55
            for route, reach, delay, size in self.sparks[:90]:
                angle = route * tau + delay * 0.08
                radius = self.size * (1.65 + reach * 2.8) * scale
                length = (22 + size * 172) * phase ** 1.2
                tail = polar(center, radius, angle)
                head = polar(center, radius + length, angle)
                color = PURPLE_WHITE if size > 0.67 else PURPLE_LIGHT
                line(layer, tail, head, tint(color, phase * (0.43 + 0.53 * size)), 1 + int(size > 0.72))
            self.draw_held(layer, self.hold_position, now, 1, scale)
            return

        age -= self.flight
        fade = exp(-age * 1.42)
        if age < 0.32:
            hot = (1 - age / 0.32) ** 2
            radius = self.size * 2.9 + 820 * age
            ring(layer, center, radius, tint(PURPLE_LIGHT, hot * 0.43), -1)
            ring(layer, center, radius * 0.61, tint(PURPLE_LIGHT, hot * 0.82), -1)
        if age < 0.86:
            self.draw_orb(layer, center, self.size * 2.8 + 138 * (1 - exp(-4.7 * age)),
                          PURPLE_LIGHT, 1, now, fade)
        self.draw_shock(layer, center, age, fade)
        if age < 0.32:
            ring(layer, center, radius * 0.29, tint(PURPLE_WHITE, hot), -1)

    def draw_infall(self, layer, center, phase):
        for route, reach, delay, size in self.sparks[:162]:
            angle = route * tau + phase * (4.3 + delay)
            distance = 6 + (100 + reach * 440) * (1 - phase) ** 1.52
            tail = polar(center, distance + 21 + size * 57, angle - 0.14)
            line(layer, tail, polar(center, distance, angle),
                 tint(PURPLE_BLUE if route < 0.5 else PURPLE_RED, (0.39 + 0.55 * size) * min(1, phase * 3)), 2)
        ring(layer, center, 4 + 16 * phase ** 2, PURPLE_WHITE, -1)

    def draw_shock(self, layer, center, age, fade):
        for i, speed in enumerate((1020, 840, 670, 490, 337)):
            radius = 12 + age * speed
            if radius < 1120:
                angle = i * 84 + age * 126
                arc(layer, center, radius, angle, angle + 294, tint(PURPLE_LIGHT, fade * (0.92 - i * 0.10)), 3)
                arc(layer, center, radius + 6, angle + 41, angle + 180, tint(PURPLE_WHITE, fade * 0.74), 2)
        for route, reach, delay, size in self.sparks:
            angle = route * tau + age * delay * 0.34
            radius = 6 + (110 + 690 * reach) * (1 - exp(-3.6 * age))
            head = polar(center, radius, angle)
            tail = polar(center, max(0, radius - (20 + size * 128) * fade), angle - 0.05)
            color = PURPLE_LIGHT if size < 0.86 else PURPLE_WHITE
            line(layer, tail, head, tint(color, fade * (0.53 + size * 0.46)), 1 + int(size > 0.76))
        if age < 0.95:
            for i in range(11):
                angle = i * tau / 11 + age * 0.47
                reach = (150 + i * 22) * min(1, age * 5)
                bolt = [polar(center, reach * j / 8, angle + sin(j * 2.38 + i) * 0.10) for j in range(9)]
                stroke(layer, bolt, tint(PURPLE_LIGHT, fade * 0.92), 4)
                stroke(layer, bolt, tint(PURPLE_WHITE, fade), 1)

    def draw_hud(self, frame, pair, preview):
        if not self.started:
            status = "Left of screen: open palm    Right: point"
        elif self.shot_at is not None:
            status = "HOLLOW PURPLE"
        elif self.fusion_at is not None:
            status = ("Open the firing hand to shoot" if self.release_armed else
                      "Keep your pointing hand visible to hold Purple")
        elif pair is None:
            status = "Bring both hands back into view"
        elif self.charge.red_at is not None:
            status = "Bring Blue and Red together"
        else:
            status = "Hold open palm and pointing hand"
        cv2.rectangle(frame, (34, 32), (306, 104), DARK, -1)
        line(frame, (34, 32), (34, 104), PURPLE_LIGHT, 2)
        label(frame, "MURASAKI", (48, 64), 0.9, PURPLE_WHITE, 2)
        label(frame, "EFFECTS PREVIEW" if preview else "LIVE CAMERA", (49, 88), 0.41, PURPLE_LIGHT)
        cv2.rectangle(frame, (34, 457), (926, 509), DARK, -1)
        label(frame, status, (48, 489), 0.46, PURPLE_WHITE)
        if self.started and self.fusion_at is None:
            for x, color, formed in (
                    (526, PURPLE_BLUE, self.charge.blue_at), (653, PURPLE_RED, self.charge.red_at)):
                line(frame, (x, 484), (x + 96, 484), tint(color, 0.28), 3)
                if formed is not None:
                    line(frame, (x, 484), (x + 96, 484), color, 3)
        label(frame, "R Reset   Esc Close", (794, 489), 0.42, PURPLE_WHITE)


# Automatic spell selection

def palm_inputs(hands):
    return [(hand.palm, hand.palm_size) for hand in hands]


def start_gesture(hands):
    if len(hands) == 2:
        if purple_sign(hands):
            return "purple", hands
        fingers = [4 if hand.pose == "open" else hand.open_fingers for hand in hands]
        if min(fingers) >= 2 and max(fingers) >= 3:
            pair = palm_pair(palm_inputs(hands))
            if pair and Chakra.can_seal(pair[1]):
                return "chakra", pair
    elif len(hands) == 1:
        hand = hands[0]
        if hand.pose == "sign":
            return "pulse", hand
    return None, None


class AutoScene:
    def __init__(self):
        self.background = None
        self.capture_at = None
        self.empty_since = None
        self.teleport_latched = False
        self.palm_twist = PalmTwist()
        self.reset()

    def reset(self):
        self.mode = None
        self.scene = None
        self.candidate = None
        self.samples = []
        self.last_palm = None
        self.range = 140
        self.touch_since = None
        self.capture_at = self.empty_since = None
        self.palm_twist.reset()
        self.pinch_pen = PinchPen()

    def capture_background(self, now):
        self.reset()
        self.capture_at = now + 2.5

    def finished(self, now):
        if self.mode == "pulse":
            cast = self.scene.cast
            return now - cast.time >= cast.lifetime if cast else not self.scene.spell.active
        if self.mode == "runes":
            effect = self.scene.effect
            return effect is not None and now - effect.started >= effect.duration
        if self.mode in ("chakra", "purple"):
            return self.scene.finished(now)
        if self.mode == "teleport":
            return now - self.scene.started >= self.scene.duration
        return False

    def select(self, now, hands, play):
        candidate, sample = start_gesture(hands)
        if candidate is None:
            if self.samples and now - self.samples[-1][0] > 0.075:
                self.candidate, self.samples = None, []
            return
        if candidate != self.candidate or not self.samples or now - self.samples[-1][0] > 0.15:
            self.candidate, self.samples = candidate, []
        self.samples.append((now, sample))
        if len(self.samples) > 96:
            self.samples.pop(0)
        began, first = self.samples[0]
        if candidate == "pulse":
            if now - began < 0.18:
                return
            self.scene = PulseScene()
            self.scene.spell.update(began, "sign", first.angle)
        elif candidate == "purple":
            if now - began < 0.20:
                return
            self.scene = PurpleScene(started=True)
        else:
            if now - began < 0.22:
                return
            self.scene = ChakraScene()
            for timestamp, pair in self.samples[:-1]:
                self.scene.chakra.update(timestamp, pair[1], pair[3])
        self.mode = candidate
        if self.mode == "pulse":
            self.last_palm = sample.palm
            self.range = max(105, 2.8 * sample.palm_size)
        self.samples.clear()

    def casting_hand(self, hands):
        if not hands:
            return None
        x, y = self.last_palm
        nearest = min(hands, key=lambda hand: hypot(hand.palm[0] - x, hand.palm[1] - y))
        if hypot(nearest.palm[0] - x, nearest.palm[1] - y) > self.range:
            return None
        self.last_palm = nearest.palm
        return nearest

    def undo(self):
        if self.mode == "runes":
            self.scene.rune.clear_stroke()

    def begin_teleport(self, now, frame, face, anchor, play):
        if face is None:
            face = FaceBox(WIDTH * 0.43, HEIGHT * 0.21, WIDTH * 0.14, HEIGHT * 0.23)
        self.scene = TeleportScene(now, frame, self.background, face, anchor)
        self.mode = "teleport"
        self.teleport_latched = True
        play("teleport")

    def draw(self, now, frame, hands, preview, play, face=None):
        if self.capture_at is not None:
            if now < self.capture_at:
                status = f"Step out of frame / {int(self.capture_at - now) + 1}"
            else:
                status = "Hold the room empty for a moment"
                if face is not None or hands:
                    self.empty_since = None
                elif self.empty_since is None:
                    self.empty_since = now
                elif now - self.empty_since >= 0.48:
                    self.background = frame.copy()
                    self.capture_at = self.empty_since = None
                    play("summon")
                    return self.draw_idle(frame, preview)
            return self.draw_idle(frame, preview, status)
        if self.finished(now):
            if (self.mode == "runes" and isinstance(self.scene.effect, PalmVortex)
                    and self.background is not None):
                center = self.scene.effect.seal.center
                anchor = (face.x + face.width / 2, face.y + face.height * 0.25) if face else center
                self.begin_teleport(now, frame, face, anchor, play)
            else:
                self.reset()
        lone = hands[0] if len(hands) == 1 else None
        touching = lone is not None and near_forehead(lone, face)
        if not touching:
            self.touch_since = None
            self.teleport_latched = False
        if self.mode == "teleport":
            output = self.scene.draw(now, frame, play).copy()
            cv2.rectangle(output, (34, 32), (260, 70), DARK, -1)
            label(output, "INSTANT SHIFT", (49, 58), 0.66, ICE, 2)
            return output
        if self.mode == "pulse" and not self.scene.cast and purple_sign(hands):
            self.select(now, hands, play)
        if touching and self.mode in (None, "pulse") and not (self.mode == "pulse" and self.scene.cast):
            if self.teleport_latched:
                return self.draw_idle(frame, preview)
            if self.background is None:
                return self.draw_idle(frame, preview, "Teleport setup / Press B, then step out of frame")
            if self.touch_since is None:
                self.touch_since = now
            if now - self.touch_since >= 0.12:
                self.begin_teleport(now, frame, face, lone.fingers, play)
                return self.scene.draw(now, frame, play)
            output = self.draw_idle(frame, preview, "Teleport / Hold both fingertips at your forehead")
            ring(output, point(*lone.fingers), 9, ICE, 2)
            return output
        if self.mode is None:
            inking = self.pinch_pen.update(now, lone)
            if lone and self.pinch_pen.since is not None:
                self.candidate, self.samples = None, []
                self.palm_twist.reset()
                if not inking:
                    output = self.draw_idle(frame, preview, "Hold pinch briefly to start drawing")
                    ring(output, point(*lone.pinch), 9, INK, 2)
                    return output
                self.mode, self.scene = "runes", RuneScene(self.pinch_pen)
                self.last_palm = lone.palm
                self.range = max(105, 2.8 * lone.palm_size)
        if self.mode is None:
            seal = self.palm_twist.update(now, lone)
            if seal:
                self.mode, self.scene = "runes", RuneScene()
                self.last_palm = lone.palm
                self.range = max(105, 2.8 * lone.palm_size)
                self.scene.rune.stage = "cast"
                self.scene.rune.seal = seal
                self.scene.effect = PalmVortex(seal, now)
                play("vortex")
            elif lone and lone.pose == "open":
                self.select(now, hands, play)
                output = self.palm_twist.draw_hint(frame, lone)
                return self.draw_idle(output, preview, "Open palm / Turn your wrist to release Vortex")
        if self.mode is None:
            self.select(now, hands, play)
            if self.mode is None:
                return self.draw_idle(frame, preview)
        if self.mode == "purple":
            gesture = hands
        elif self.mode == "chakra":
            gesture = palm_inputs(hands)
        else:
            gesture = self.casting_hand(hands)
        return self.scene.draw(now, frame, gesture, preview, play)

    def draw_idle(self, frame, preview, status=None):
        output = frame.copy()
        cv2.rectangle(output, (34, 32), (316, 105), DARK, -1)
        line(output, (34, 32), (34, 105), ICE, 2)
        label(output, "ARCANE SIGNAL", (49, 62), 0.78, WHITE, 2)
        label(output, "AUTO SPELLS" if not preview else "AUTO SPELLS / PREVIEW", (50, 89), 0.44, ICE)
        options = (
            ("runes", "RUNE SEALS", "Hold pinch, then draw", INK),
            ("pulse", "PULSE", "Hold up two fingers", GOLD),
            ("chakra", "CHAKRA", "Bring two open palms close", CYAN),
            ("purple", "MURASAKI", "Left palm / right point", PURPLE_LIGHT),
        )
        for i, (mode, title, instruction, color) in enumerate(options):
            x = 34 + 223 * i
            cv2.rectangle(output, (x, 422), (x + 213, 487), DARK, -1)
            pending = self.pinch_pen.since is not None if mode == "runes" else self.candidate == mode
            line(output, (x, 422), (x + 213, 422), tint(color, 1 if pending else 0.42), 2)
            label(output, title, (x + 14, 447), 0.49, color)
            label(output, instruction, (x + 14, 471), 0.42)
        if status is None:
            status = ("Open palm + twist: Vortex   /   Two fingers to forehead: Teleport" if self.background is not None
                      else "Open palm + twist: Vortex   /   B: Capture room for Teleport")
        cv2.rectangle(output, (34, 494), (920, 523), DARK, -1)
        label(output, status, (48, 513), 0.43, ICE)
        label(output, "R Reset    Esc Close", (756, 513), 0.41)
        return output


# Application entry points

def rune_demo_hand(now):
    phase = now % 8.6
    center = (484, 262)
    example = int(now // 8.6) % 3
    direction = -1 if example == 1 else 1
    start_angle = -tau / 4
    if 0.27 <= phase < 2.02:
        t = max(0, (phase - 0.53) / 1.49)
        angle = start_angle + 0.91 * tau * t
        radius = 104 + 3 * sin(angle * 3)
    elif 2.34 <= phase < 4.08:
        t = max(0, (phase - 2.60) / 1.48)
        if example == 2:
            vertices = [(444, 209), (535, 215), (431, 281), (518, 302)]
            segment = min(2, int(t * 3))
            fraction = t * 3 - segment
            tip = tuple(a + (b - a) * fraction for a, b in zip(vertices[segment], vertices[segment + 1]))
            return Hand("other", 0, tip, center, tip, pinch=tip, pinch_gap=0.10)
        angle = start_angle + direction * 0.64 * tau * t
        radius = 81 - 67 * t
    else:
        pose = "open" if 4.60 <= phase <= 5.29 else "other"
        return Hand(pose, 0, center, center, (617, 327))
    tip = (center[0] + radius * cos(angle), center[1] + radius * sin(angle))
    return Hand("other", 0, tip, center, tip, pinch=tip, pinch_gap=0.10)


AUTO_DEMO_PERIOD = 64.1
TELEPORT_DEMO_START = 51.2
PALM_DEMO_START = 41.2
PURPLE_DEMO_START = 56.1
DEMO_FACE = FaceBox(425, 138, 111, 124)


def purple_demo_hands(now):
    phase = now % 8.0
    if not 0.19 <= phase < 5.20:
        return ()
    merge = min(1, max(0, (phase - 1.57) / 0.58))
    gap = 294 - merge * 199
    y = 285 - 13 * merge
    sway = 51 * sin(max(0, phase - 2.80) * 1.35)
    x = 480 + gap / 2 + sway
    pose = "open" if phase >= 4.65 else "point"
    right = Hand(pose, 0, (x, y - 36), (x, y), index=(x, y - 48), palm_size=67,
                 open_fingers=4 if pose == "open" else 1, handedness="Right")
    if phase >= 2.50:
        return (right,)
    left_x = 480 - gap / 2
    return (Hand("open", 0, (left_x, y - 35), (left_x, y), palm_size=67,
                 open_fingers=4, handedness="Left"), right)


def auto_demo_hands(now):
    phase = now % AUTO_DEMO_PERIOD
    if phase >= PURPLE_DEMO_START:
        return purple_demo_hands(phase - PURPLE_DEMO_START)
    if phase < 8.8:
        hand = demo_hand(phase)
    elif phase < 34.6:
        hand = rune_demo_hand(phase - 8.8)
    elif phase < PALM_DEMO_START:
        return tuple(Hand("open", 0, position, position, palm_size=scale)
                     for position, scale in demo_palms(phase - 34.6))
    elif phase < TELEPORT_DEMO_START:
        local = phase - PALM_DEMO_START
        if 0.25 <= local < 1.20:
            palm = (481, 265)
            angle = radians(78 * min(1, max(0, (local - 0.40) / 0.60)))
            hand = Hand("open", angle, palm, palm, palm_size=80, open_fingers=4)
        else:
            hand = None
    elif TELEPORT_DEMO_START + 0.40 <= phase <= TELEPORT_DEMO_START + 0.95:
        hand = Hand("sign", 0, (486, 167), (543, 247), (479, 164), palm_size=70)
    else:
        hand = None
    return (hand,) if hand else ()


def teleport_demo_frame(backdrop):
    frame = backdrop.copy()
    cv2.ellipse(frame, (480, 548), (124, 282), 0, 0, 360, (96, 66, 106), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (480, 201), (63, 81), 0, 0, 360, (149, 125, 164), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (480, 161), (63, 46), 0, 0, 180, (78, 53, 89), 7, cv2.LINE_AA)
    line(frame, (562, 355), (538, 232), (182, 147, 184), 22)
    line(frame, (538, 232), (488, 166), (182, 147, 184), 12)
    line(frame, (547, 230), (497, 161), (182, 147, 184), 8)
    return frame


def run(preview, camera_index, mode="auto"):
    camera = None
    tracker = None
    detector = None
    face = None
    face_checked = face_seen = -1.0
    if mode == "auto":
        scene, preview_input, window = AutoScene(), auto_demo_hands, WINDOW
    elif mode == "purple":
        scene, preview_input, window = PurpleScene(), purple_demo_hands, "Murasaki"
    elif mode == "chakra":
        scene, preview_input, window = ChakraScene(), demo_palms, "Chakra Surge"
    elif mode == "pulse":
        scene, preview_input, window = PulseScene(), demo_hand, WINDOW
    else:
        scene, preview_input, window = RuneScene(), rune_demo_hand, "Rune Seals"
    try:
        if preview:
            backdrop = make_backdrop()
            if mode == "auto":
                scene.background = backdrop
                demo_person = teleport_demo_frame(backdrop)
        else:
            tracker = make_tracker(2 if mode in ("chakra", "purple", "auto") else 1)
            if mode == "auto":
                detector = make_face_detector()
            camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError("Could not open the camera. Try --camera 1 or --preview.")

        with TemporaryDirectory() as audio:
            prepare_sounds(audio)
            try:
                import winsound
            except ImportError:
                winsound = None

            def play(name):
                if winsound:
                    try:
                        winsound.PlaySound(str(Path(audio) / f"{name}.wav"),
                                           winsound.SND_ASYNC | winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                    except RuntimeError:
                        pass

            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            start = monotonic()
            previous_timestamp = -1
            try:
                while True:
                    if camera is not None:
                        ok, raw = camera.read()
                        if not ok:
                            raise RuntimeError("The camera stopped sending frames.")
                        frame = camera_frame(raw)
                        now = monotonic() - start
                        previous_timestamp = max(previous_timestamp + 1, round(now * 1000))
                        hands = track_hands(tracker, frame, previous_timestamp)
                        if mode == "auto":
                            gesture = hands
                            if now - face_checked >= 0.072:
                                found = track_face(detector, frame, previous_timestamp)
                                face_checked = now
                                if found:
                                    face, face_seen = found, now
                            if now - face_seen > 0.32:
                                face = None
                        elif mode == "chakra":
                            gesture = palm_inputs(hands)
                        elif mode == "purple":
                            gesture = hands
                        else:
                            gesture = hands[0] if hands else None
                        frame = cv2.convertScaleAbs(frame, alpha=0.66)
                    else:
                        now = monotonic() - start
                        frame, gesture = backdrop, preview_input(now)
                        if mode == "auto":
                            phase = now % AUTO_DEMO_PERIOD
                            in_teleport_demo = PALM_DEMO_START <= phase < PURPLE_DEMO_START
                            frame = demo_person if in_teleport_demo else backdrop
                            face = DEMO_FACE if in_teleport_demo else None
                    output = (scene.draw(now, frame, gesture, preview, play, face=face) if mode == "auto"
                              else scene.draw(now, frame, gesture, preview, play))
                    cv2.imshow(window, output)
                    key = cv2.waitKey(15) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("r") and mode != "pulse":
                        scene.reset()
                        if preview:
                            start = monotonic()
                    if key == 8 and mode == "runes":
                        scene.rune.clear_stroke()
                    if key == 8 and mode == "auto":
                        scene.undo()
                    if key == ord("b") and mode == "auto" and not preview:
                        scene.capture_background(now)
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
            finally:
                if winsound:
                    winsound.PlaySound(None, 0)
    finally:
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.close()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Cast a spell using your webcam.")
    parser.add_argument("--preview", action="store_true", help="Preview the spell without a camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--pulse", dest="mode", action="store_const", const="pulse",
                       help="Open the original two-finger pulse demo")
    modes.add_argument("--chakra", dest="mode", action="store_const", const="chakra",
                       help="Open the two-hand Chakra Surge demo")
    modes.add_argument("--runes", dest="mode", action="store_const", const="runes",
                       help="Open Rune Seals without automatic spell selection")
    modes.add_argument("--purple", dest="mode", action="store_const", const="purple",
                       help="Open the two-hand Murasaki demo")
    parser.set_defaults(mode="auto")
    args = parser.parse_args()
    try:
        run(args.preview, args.camera, args.mode)
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        parser.exit(1, f"Arcane Signal: {error}\n")


if __name__ == "__main__":
    main()
