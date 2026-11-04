# Wilson-Day scoreboard regression crops

Twelve original-resolution 810 x 60 score/name-bar crops from the locally supplied
Kyren Wilson vs Ryan Day source. `cases.json` records source seconds, both white
points values, and the active player (0 left, 1 right, null unreadable).
Reconstruct each crop at `[550:610, 230:1040]` in a black 1280 x 720 image.

Source SHA-256: `8233f54d87e238f5f0c9570c8a521108c589f18e308c2a1f23b9ff5550bc934f`.

These cover dim antialiased glyphs, the red break counter, yellow-panel edge
bleed, and the white turn dots at the outer ends of the name panels. The 6959s
sample deliberately retains an unreadable turn dot. Confidence and digit
ambiguity gates remain enabled; null is preferable to guessing a turn.
