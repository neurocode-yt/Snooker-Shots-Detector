# Score digit samples

`score_digits.npz` contains 515 raster digit templates for classical shape
matching: digits, normalized images, aspect ratios and hole counts. These are
fixed rendered samples of OpenCV Hershey and Arial, Segoe UI and Calibri digits,
at several sizes and stroke widths, plus 35 manually verified digit glyphs from
the Lilley–Liu and Holt–Burns broadcast overlays. They contain no neural model or video frames.
The packaged samples give the same recognition templates on Windows and Linux.

The score reader checks glyph shape, holes, confidence and the margin between
candidate digits, then requires repeated stable readings. Frame counters are
excluded by overlay geometry. The cache fingerprint includes the sample file.
