"""
Background subtraction — MOG2 (Mixture of Gaussians v2).

Theory:
  Each pixel's intensity history is modelled as a mixture of K Gaussians
  (typically 3–5). At each frame, pixels whose value falls outside the
  expected distribution are marked as foreground (value 255 in the mask).
  The model adapts continuously, so slow illumination changes are absorbed
  into the background model over time.

  Reference: Zivkovic, Z. (2004). "Improved adaptive Gaussian mixture model
  for background subtraction." ICPR.

Why MOG2 over frame-difference?
  Frame differencing only captures pixels that *changed between two frames*,
  so a slowly-moving vehicle leaves gaps.  MOG2 maintains a history of
  N frames, giving much cleaner foreground masks for real traffic scenes.
"""
import cv2


def create_subtractor(history: int = 300, var_threshold: float = 50.0) -> cv2.BackgroundSubtractorMOG2:
    """
    Create a MOG2 background subtractor.

    Args:
        history       : how many past frames contribute to the background model.
                        Larger = slower to adapt (good for static camera).
        var_threshold : Mahalanobis-distance² threshold.
                        Lower values → more pixels labelled foreground (more sensitive).
                        Higher values → only large changes trigger foreground.

    Returns:
        cv2.BackgroundSubtractorMOG2
    """
    return cv2.createBackgroundSubtractorMOG2(
        history        = history,
        varThreshold   = var_threshold,
        detectShadows  = False,   # shadows cause false positives; disable them
    )


def apply_subtractor(
    subtractor: cv2.BackgroundSubtractorMOG2,
    frame,
    learning_rate: float = -1.0,
):
    """
    Apply MOG2 to one frame and return the foreground mask.

    Args:
        subtractor    : MOG2 object created by create_subtractor()
        frame         : BGR frame (uint8 H×W×3)
        learning_rate : -1 = automatic (OpenCV decides per history).
                         0 = freeze model (useful to 'lock' the background).
                         0..1 = manual update speed.

    Returns:
        fg_mask : uint8 binary mask, 255 = foreground pixel, 0 = background
    """
    return subtractor.apply(frame, learningRate=learning_rate)
