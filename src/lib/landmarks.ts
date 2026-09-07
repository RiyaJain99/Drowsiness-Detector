/**
 * Landmark indices into MediaPipe's 478-point face mesh.
 *
 * Eye points are ordered p1..p6 as the eye-aspect-ratio formula expects:
 * outer corner, two upper lid points, inner corner, two lower lid points.
 */

export const RIGHT_EYE = [33, 160, 158, 133, 153, 144] as const;
export const LEFT_EYE = [362, 385, 387, 263, 373, 380] as const;

/** Inner lip, top and bottom — vertical mouth opening. */
export const MOUTH_VERTICAL = [13, 14] as const;
/** Mouth corners — horizontal reference so the ratio is scale-invariant. */
export const MOUTH_HORIZONTAL = [61, 291] as const;

/** Head-pose reference points. */
export const FOREHEAD = 10;
export const CHIN = 152;
export const NOSE_TIP = 1;
/** Outer eye corners — the interocular axis, which pitch rotation leaves unchanged. */
export const EYE_OUTER = [33, 263] as const;

/** Full lid contours, used only for drawing the overlay. */
export const EYE_CONTOURS = {
  right: [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
  left: [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
} as const;

/** Jaw outline, for the HUD face box. */
export const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
] as const;
