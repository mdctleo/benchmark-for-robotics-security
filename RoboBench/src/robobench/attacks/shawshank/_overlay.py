"""Shawshank's text projection: paint the sign onto the frame.

``project_text_on_image`` is ported from the upstream Shawshank code: it renders
the text onto a canvas, warps it with a perspective transform so it sits in the
scene, and alpha-blends it faintly over the frame. ``decode`` / ``encode`` convert
between the artifact's image bytes and the OpenCV arrays this works on.

Upstream captured its scenes at a fixed 640x480 and resized every frame to that
size before placing the sign, so its sign coordinates are calibrated to 640x480.
RoboBench frames come at their own resolutions, so instead of resizing the frame
we scale the sign's placement to the frame's own size against that 640x480
reference: the frame keeps its native resolution and aspect ratio, and on a
640x480 frame the placement is identical to upstream.

The overlaid frame is re-encoded in the same container as the input frame
(``source_ext``), so it is served to the planner in the base's native format
(PNG stays PNG, JPEG stays JPEG) rather than recompressed.
"""

import textwrap

import cv2
import numpy as np


def source_ext(raw: bytes) -> str:
    """OpenCV ``imencode`` extension matching the input's container.

    Mirrors ``robobench.models.detect_media_type``: PNG magic -> ``.png``, JPEG
    magic -> ``.jpg``, anything else -> ``.png`` (lossless).
    """
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if raw[:3] == b"\xff\xd8\xff":
        return ".jpg"
    return ".png"


def decode(raw: bytes) -> np.ndarray:
    """Encoded image bytes -> OpenCV BGR array."""
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image bytes")
    return image


def encode(image_rgb: np.ndarray, ext: str = ".png") -> bytes:
    """RGB array (as ``project_text_on_image`` returns) -> bytes in ``ext``'s format."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError("could not encode image")
    return buffer.tobytes()


def project_text_on_image(image, text, text_position, font_scale=2, font_thickness=7,
                          alpha=0.2, reference_size=(640, 480), line_height=100,
                          max_line_width=700):
    target_image = image.copy()
    height, width = target_image.shape[:2]

    # Scale the sign's placement to this frame's size. ``reference_size`` is the
    # resolution upstream's coordinates were tuned for (its 640x480 capture size);
    # on a frame of that size the scale is 1 and the placement matches upstream.
    ref_width, ref_height = reference_size
    scale_x, scale_y = width / ref_width, height / ref_height

    text_image = np.zeros((2000, 2000, 4), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    y_pos = text_position[1]

    wrapped_text = textwrap.fill(text, width=max_line_width)

    lines = wrapped_text.split("\n")
    for line in lines:
        cv2.putText(text_image, line, (text_position[0], y_pos), font, font_scale,
                    color, font_thickness, lineType=cv2.LINE_AA)
        y_pos += line_height

    pts1 = np.float32([[100 * scale_x, 100 * scale_y], [700 * scale_x, 100 * scale_y],
                       [100 * scale_x, 600 * scale_y], [700 * scale_x, 600 * scale_y]])
    pts2 = np.float32([[0, 0], [text_image.shape[1], 0], [0, text_image.shape[0]],
                       [text_image.shape[1], text_image.shape[0]]])

    matrix = cv2.getPerspectiveTransform(pts2, pts1)

    projected = cv2.warpPerspective(text_image, matrix, (width, height))

    for c in range(0, 3):
        target_image[:, :, c] = ((1 - alpha) * target_image[:, :, c]
                                 + alpha * projected[:, :, c])

    image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    return image_rgb
