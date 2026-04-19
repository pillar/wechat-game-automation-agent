"""macOS Vision-Framework-based OCR for text-labeled UI buttons.

Used to override VLM coords for buttons with known Chinese text labels
(升级 / 确认 / 领取 / 前往 / ...). The VLM tends to hallucinate Y by ~100 px
for these buttons; OCR gives us the real bbox in pixel space.

Zero new deps: pyobjc (already a requirement) exposes the Vision framework.

API:
    find_text(pil_image, keywords=None) -> List[Dict]
        Returns a list of detections in pixel space (top-origin):
            {text, x, y, w, h, cx, cy, confidence}
        If keywords is provided, only detections whose text contains
        any keyword (substring match) are returned.

Failure modes:
    - Returns [] if Vision framework is unavailable (non-macOS) or the
      request errors. Callers must fall back to VLM coords.
    - Confidence thresholding is the caller's responsibility.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

logger = logging.getLogger(__name__)

_VISION_OK: Optional[bool] = None
_Vision = None
_Quartz = None
_CoreFoundation = None


def _ensure_vision() -> bool:
    global _VISION_OK, _Vision, _Quartz, _CoreFoundation
    if _VISION_OK is not None:
        return _VISION_OK
    try:
        import Vision  # type: ignore
        import Quartz  # type: ignore
        import CoreFoundation  # type: ignore
        _Vision = Vision
        _Quartz = Quartz
        _CoreFoundation = CoreFoundation
        _VISION_OK = True
    except Exception as e:
        logger.debug(f"[OCR] Vision framework unavailable: {e}")
        _VISION_OK = False
    return _VISION_OK


def _pil_to_cgimage(img: Image.Image):
    """Convert PIL image to CGImage via PNG bytes → CFData → CGImageSource."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()
    data = _CoreFoundation.CFDataCreate(None, png_bytes, len(png_bytes))
    src = _Quartz.CGImageSourceCreateWithData(data, None)
    if src is None:
        return None
    return _Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def find_text(
    pil_image: Image.Image,
    keywords: Optional[Sequence[str]] = None,
    min_confidence: float = 0.3,
) -> List[Dict[str, Any]]:
    """Run macOS Vision OCR on a PIL image, return detections in pixel space.

    Args:
        pil_image: PIL Image to OCR.
        keywords: Optional substring filter; only detections containing any
            keyword in their recognized text are returned.
        min_confidence: Drop detections below this confidence.

    Returns:
        List of dicts with keys: text, x, y, w, h, cx, cy, confidence.
        Pixel coordinates are top-origin (image space).
        Empty list on any failure.
    """
    if not _ensure_vision():
        return []
    try:
        cg = _pil_to_cgimage(pil_image)
        if cg is None:
            return []
        w_img, h_img = pil_image.size

        req = _Vision.VNRecognizeTextRequest.alloc().init()
        try:
            req.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en-US"])
        except Exception:
            pass
        try:
            req.setUsesLanguageCorrection_(False)
        except Exception:
            pass
        try:
            req.setRecognitionLevel_(_Vision.VNRequestTextRecognitionLevelAccurate)
        except Exception:
            pass

        handler = _Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            logger.debug(f"[OCR] performRequests failed: {err}")
            return []

        results = req.results() or []
        out: List[Dict[str, Any]] = []
        for obs in results:
            try:
                cand = obs.topCandidates_(1)
                if not cand or len(cand) == 0:
                    continue
                top = cand[0]
                text = str(top.string())
                conf = float(top.confidence())
            except Exception:
                continue
            if conf < min_confidence:
                continue
            if keywords and not any(k in text for k in keywords):
                continue
            # Vision bbox is normalized with BOTTOM-origin Y.
            try:
                bb = obs.boundingBox()
                nx, ny, nw, nh = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
            except Exception:
                continue
            px = int(nx * w_img)
            pw = int(nw * w_img)
            ph = int(nh * h_img)
            # Flip Y: top = h_img - (ny + nh) * h_img
            py = int(h_img - (ny + nh) * h_img)
            out.append({
                "text": text,
                "x": px,
                "y": py,
                "w": pw,
                "h": ph,
                "cx": px + pw // 2,
                "cy": py + ph // 2,
                "confidence": conf,
            })
        return out
    except Exception as e:
        logger.debug(f"[OCR] exception: {e}")
        return []


def find_nearest(
    pil_image: Image.Image,
    keywords: Sequence[str],
    hint_x: int,
    hint_y: int,
    min_confidence: float = 0.3,
    max_dist: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Find the OCR detection matching `keywords` closest to (hint_x, hint_y).

    Returns the single best match or None. If multiple detections match,
    picks the one whose center is nearest to the hint. If max_dist is set,
    rejects candidates farther than that in pixels.
    """
    hits = find_text(pil_image, keywords=keywords, min_confidence=min_confidence)
    if not hits:
        return None
    best = None
    best_d = float("inf")
    for h in hits:
        d = ((h["cx"] - hint_x) ** 2 + (h["cy"] - hint_y) ** 2) ** 0.5
        if max_dist is not None and d > max_dist:
            continue
        if d < best_d:
            best_d = d
            best = h
    if best is not None:
        best = dict(best)
        best["dist_to_hint"] = best_d
    return best
