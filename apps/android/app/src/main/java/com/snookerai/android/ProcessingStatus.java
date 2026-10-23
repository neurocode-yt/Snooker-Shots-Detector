package com.snookerai.android;

import org.json.JSONArray;
import org.json.JSONObject;

/** A single notification describes the same work list as the editor. */
final class ProcessingStatus {
    final boolean busy, pending;
    final int percent;
    final String title, text, detail;

    ProcessingStatus(boolean busy, boolean pending, int percent, String title, String text, String detail) {
        this.busy = busy; this.pending = pending; this.percent = percent;
        this.title = title; this.text = text; this.detail = detail;
    }

    static ProcessingStatus read(JSONArray items, String nativeWork) {
        JSONObject selected = null;
        int active = 0;
        for (int i = 0; i < items.length(); i++) {
            JSONObject item = items.optJSONObject(i);
            if (item == null) continue;
            String state = item.optString("status");
            if (state.equals("paused") || state.equals("interrupted")) continue;
            active++;
            // A queued match must not hide the match currently being processed.
            if (selected == null || queued(selected) && !queued(item)) selected = item;
        }
        if (selected == null && nativeWork == null) {
            boolean saved = items.length() > 0;
            return new ProcessingStatus(false, saved, -1, "Snooker App",
                saved ? "Work paused · checkpoints saved" : "Ready · Choose a video to begin",
                saved ? "Open the app to continue saved work." : "All processing runs on this phone.");
        }
        if (selected == null) return new ProcessingStatus(true, true, -1,
            "Snooker App · processing", nativeWork, "You can close the editor. Processing continues in the background.");
        double progress = selected.optDouble("progress", Double.NaN);
        int percent = Double.isFinite(progress) ? (int)Math.round(Math.max(0, Math.min(1, progress)) * 100) : -1;
        String message = selected.optString("message", selected.optString("status", "Processing"));
        String path = selected.optString("source_path", "").replace('\\', '/');
        String name = path.substring(path.lastIndexOf('/') + 1);
        String detail = (name.isEmpty() ? "" : name + "\n") + message + (percent >= 0 ? " · " + percent + "%" : "")
            + (active > 1 ? "\n" + active + " tasks running or queued." : "")
            + "\nYou can close the editor. Processing continues in the background.";
        return new ProcessingStatus(true, true, percent, "Snooker App · processing",
            message + (percent >= 0 ? " · " + percent + "%" : ""), detail);
    }

    private static boolean queued(JSONObject item) {
        return item.optBoolean("queued") || "queued".equals(item.optString("status"));
    }

    String key() { return title + text + detail + percent; }
}
