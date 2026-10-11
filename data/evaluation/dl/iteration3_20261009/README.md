# Independent Robertson–Un-Nooh source review

`blind-holdout-labels.json` freezes labels before any detector predictions. Its SHA-256 is `692b007490c32cc800994f34ede976c588521c6045b43c6f800ea5fc99b2339d`. Checkpoint-specific test manifests must be separate copies, with the checkpoint hash and compatible feature path pinned before inference. Preserve these source annotations.

The source is Jimmy Robertson versus Thepchaiya Un-Nooh, 2022 Championship League, SHA-256 `7bab021b21336f00d3fd6c4271d3bf947dd99c433f4bb99e17d6f56f4fb1c718`. The raw match lasts **3017.758186 seconds**. This test covers the declared absolute-source interval **0–620 seconds**, containing the complete first frame from break-off through concession plus the inter-frame transition. It is not a complete-match test.

The review used all **621** chronological one-second native-PTS samples on **39** overview pages, then 8 fps source-only refinement of every live contact. The dense review includes **33** contact pages and **3** extra pages for two uncertain contacts. There are **32 live strokes**, **11 referee handling intervals**, **5 match replay blocks**, and an archive/promotion interval at 603–620 seconds. Replay and handling use the evaluated `labels` schema, including explicit negatives outside half-second uncertainty margins.

Source refinement corrected the coarse safety timing to 194.16–194.28 seconds and resolved an initially presumed broadcast-occluded Robertson red: the live camera returns with a stationary cue ball at 507.28, before it moves at 507.52. These corrections occurred before freezing and before predictions.

Contact uncertainty is 0.12–0.28 seconds. The 32 editorial clip endpoints are weak preferences at weight 0.35 with one-second uncertainty, and do not establish physical ball stops. The final shot cuts to a player view before complete settling; its preferred endpoint retains the visible outcome. Labels are AI visual review and **not human approved**.

`source-review-audit.json` verifies source identity, complete declared-segment overview coverage, contact support at the feature cadence, schema validity, and absence of live contacts within positive replay labels. This audit contains no model inference or accuracy claim.
