# Virtual Device App UI Plan (Next.js + shadcn/ui)

## Goal

Build a Next.js App Router UI for a virtual device app (Voicemod-inspired) with three main features:

1. Prestream whitelist enrollment (face capture, whitelisting, blurred faces for non-whitelisted)
2. Streaming (audio/video/screenshare, real-time blurring of non-whitelisted faces/QR/PII)
3. Post-stream safety score calculation

## Navigation Flow

- `/enrollment`: Whitelist enrollment (camera, form, preview)
- `/stream`: Streaming interface (controls, video/audio/screenshare, overlays)
- `/summary`: Post-stream safety score and summary

## Atomic UI Components (shadcn/ui)

- EnrollmentDialog, CameraCapture, FacePreview, WhitelistStatusBadge
- StreamControls, VideoPreview, OverlayBlur, ScreensharePreview
- SafetyScoreCard, SummaryDialog

## Modular File Structure

- app/enrollment/
- app/stream/
- app/summary/
- components/ui/
- components/enrollment/
- components/stream/
- components/summary/
- lib/

## Implementation Notes

- All UI components use shadcn/ui for consistency and rapid development.
- Each feature is isolated in its own directory for maintainability.
- Utility functions (e.g., blur effect) go in lib/.
