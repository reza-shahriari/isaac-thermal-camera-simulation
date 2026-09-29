"""Publishing the four outputs on ROS 2 topics (M10.10b, §13.4 line 1027, §13.6 step 4).

**rclpy is not a dependency of this project and is not installed here.** ROS 2 is a system-level
install, not a pip package this repository can pull in, so everything in this module that can be
built and checked without it *is* -- the messages, the intrinsics, the encodings, the stamp -- and
only the final publish needs a running ROS graph. That split is deliberate: the part most likely to
be wrong is the arithmetic, and the arithmetic is testable on any machine.

**What the intrinsics are, and why they are derived rather than authored.** A pinhole camera's
focal length in pixels is ``fx = f / p`` -- the focal length over the pixel pitch -- and the
principal point is the array centre in the project's pixel-edge convention, ``(width/2,
height/2)``, *not* ``((width - 1)/2, …)``. The half-pixel difference is invisible in a preview and
is a systematic half-pixel bias in anything that triangulates. Both come from the sensor config, so
a camera and its `CameraInfo` cannot disagree.

**Encodings follow non-negotiable #2.** Apparent temperature goes out as ``32FC1`` and never as
``16UC1``: at 300 K a float16 grid is 0.25 K coarse, five times a 50 mK NETD, and a consumer that
received kelvin in half precision would be reading a camera five times worse than the one that
produced it. The raw ADC plane goes as ``mono16`` because it *is* integer counts, and the display
branch as ``rgba8`` because that is what the ISP produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorConfig
from irsim.optics.projection import Intrinsics

__all__ = [
    "Encoding",
    "TOPIC_ENCODINGS",
    "CameraInfoMessage",
    "ImageMessage",
    "camera_info",
    "image_message",
    "frame_messages",
    "publish_frame",
]

Encoding = Literal["32FC1", "mono16", "rgba8"]

#: output name -> (topic suffix, encoding). The four §12.2 outputs, each in the form that does not
#: lose information: kelvin in float32, ADC counts as integers, the picture as bytes.
TOPIC_ENCODINGS: dict[str, tuple[str, Encoding]] = {
    "apparent_t": ("apparent_t", "32FC1"),
    "radiance": ("radiance", "32FC1"),
    "dn16": ("image_raw", "mono16"),
    "display8": ("image_color", "rgba8"),
}


@dataclass(frozen=True)
class CameraInfoMessage:
    """The `sensor_msgs/CameraInfo` fields, as plain data so they can be checked without ROS."""

    width: int
    height: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]  # row-major 3x3
    frame_id: str
    stamp_ns: int

    @property
    def fx(self) -> float:
        return self.k[0]

    @property
    def fy(self) -> float:
        return self.k[4]

    @property
    def cx(self) -> float:
        return self.k[2]

    @property
    def cy(self) -> float:
        return self.k[5]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageMessage:
    """A `sensor_msgs/Image` in the same spirit: the array, its encoding, and the header."""

    topic: str
    encoding: Encoding
    height: int
    width: int
    step: int
    data: NDArray[Any]
    frame_id: str
    stamp_ns: int
    is_bigendian: bool = False

    def __post_init__(self) -> None:
        expected = {
            "32FC1": np.float32,
            "mono16": np.uint16,
            "rgba8": np.uint8,
        }[self.encoding]
        if self.data.dtype != expected:
            raise TypeError(
                f"{self.topic}: {self.encoding} carries {np.dtype(expected).name}, got "
                f"{self.data.dtype}. Encodings are not a formatting choice -- 32FC1 exists here "
                "because float16 kelvin is 0.25 K coarse at 300 K (non-negotiable #2)"
            )


def _stamp_ns(t_s: float) -> int:
    """Sim time in integer nanoseconds. Integer, because a ROS stamp is (sec, nanosec) and a float
    second at hour scale has already lost sub-microsecond resolution."""
    return int(round(float(t_s) * 1e9))


def camera_info(
    sensor: SensorConfig, *, t_s: float, frame_id: str = "ir_camera"
) -> CameraInfoMessage:
    """`CameraInfo` from the sensor's own optics and FPA blocks (M10.9a's intrinsics)."""
    spec = sensor.sensor
    height, width = spec.fpa_shape
    # SC.33: the calibrated pinhole and lens when the file carries one, else the designed ones
    intr = Intrinsics.from_sensor(spec, 1)
    lens = spec.effective_distortion
    return CameraInfoMessage(
        width=int(width),
        height=int(height),
        distortion_model=str(lens.model),
        d=tuple(float(c) for c in lens.coeffs),
        k=(intr.fx_px, 0.0, intr.cx_px, 0.0, intr.fy_px, intr.cy_px, 0.0, 0.0, 1.0),
        frame_id=frame_id,
        stamp_ns=_stamp_ns(t_s),
    )


def image_message(
    name: str,
    array: Any,
    *,
    t_s: float,
    namespace: str = "/ir",
    frame_id: str = "ir_camera",
) -> ImageMessage:
    """One output plane as an `Image`. Refuses a plane whose dtype is not the encoding's."""
    if name not in TOPIC_ENCODINGS:
        raise KeyError(f"unknown output {name!r}; known: {sorted(TOPIC_ENCODINGS)}")
    suffix, encoding = TOPIC_ENCODINGS[name]
    data = np.asarray(array)
    if data.dtype == np.float64:
        data = data.astype(np.float32)  # narrowing to the wire form is fine; float16 is not
    height, width = int(data.shape[0]), int(data.shape[1])
    channels = int(data.shape[2]) if data.ndim == 3 else 1
    return ImageMessage(
        topic=f"{namespace.rstrip('/')}/{suffix}",
        encoding=encoding,
        height=height,
        width=width,
        step=width * channels * data.dtype.itemsize,
        data=data,
        frame_id=frame_id,
        stamp_ns=_stamp_ns(t_s),
    )


def frame_messages(
    outputs: Any,
    sensor: SensorConfig,
    *,
    t_s: float,
    namespace: str = "/ir",
    frame_id: str = "ir_camera",
) -> tuple[list[ImageMessage], CameraInfoMessage]:
    """Every output the camera actually produced, plus one `CameraInfo`.

    An output the config switched off is ``None`` and is **skipped**, not published as zeros: a
    subscriber that received a black `apparent_t` topic would have no way to tell it from a scene
    at absolute zero.
    """
    messages = []
    for name in TOPIC_ENCODINGS:
        value = getattr(outputs, name, None)
        if value is not None:
            messages.append(
                image_message(name, value, t_s=t_s, namespace=namespace, frame_id=frame_id)
            )
    return messages, camera_info(sensor, t_s=t_s, frame_id=frame_id)


def publish_frame(node: Any, messages: list[ImageMessage], info: CameraInfoMessage) -> None:
    """Push one frame onto a live ROS 2 graph. **The only part that needs rclpy.**

    ``node`` is expected to expose a ``publisher(topic, encoding)`` factory; the import is lazy and
    the error names what is missing, because ROS 2 is a system install and not something this
    repository can add for you.
    """
    try:
        import rclpy  # noqa: F401
        from sensor_msgs.msg import CameraInfo, Image  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ROS 2 (rclpy and sensor_msgs) is not available in this environment. It is a "
            "system-level install, not a pip dependency this project can pull in; source a ROS 2 "
            "setup script and re-run. Everything except this call is testable without it."
        ) from error
    for message in messages:
        node.publisher(message.topic, message.encoding).publish(message)
    node.publisher(f"{messages[0].topic.rsplit('/', 1)[0]}/camera_info", "info").publish(info)
