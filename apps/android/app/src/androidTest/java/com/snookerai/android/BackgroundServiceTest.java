package com.snookerai.android;

import android.app.*;
import android.content.*;
import android.os.SystemClock;
import android.test.InstrumentationTestCase;
import org.json.*;

/** Checks real service lifetime and notification contents, independently of the WebView. */
public class BackgroundServiceTest extends InstrumentationTestCase {
    public void testProgressSelectsRunningWorkAndPreservesPausedState() throws Exception {
        JSONArray items = new JSONArray()
            .put(new JSONObject().put("status", "paused").put("message", "Saved earlier"))
            .put(new JSONObject().put("status", "pending").put("queued", true).put("message", "Waiting"))
            .put(new JSONObject().put("status", "analyzing").put("progress", .375)
                .put("source_path", "/storage/match.mp4").put("message", "Confirming shots"));
        ProcessingStatus state = ProcessingStatus.read(items, null);
        assertTrue(state.busy);
        assertEquals(38, state.percent);
        assertTrue(state.text.contains("Confirming shots"));
        assertTrue(state.detail.contains("match.mp4"));
        assertTrue(state.detail.contains("38%"));
        assertTrue(state.detail.contains("2 tasks"));
        state = ProcessingStatus.read(new JSONArray().put(items.get(0)), null);
        assertFalse(state.busy);
        assertTrue(state.pending);
        assertTrue(state.text.contains("paused"));
        state = ProcessingStatus.read(new JSONArray(), "Saving exported video…");
        assertTrue(state.busy);
        assertEquals(-1, state.percent);
        assertFalse(ProcessingStatus.read(new JSONArray(), null).pending);
    }

    public void testServiceSurvivesTaskRemovalAndStopsWhenIdle() throws Exception {
        Context context = getInstrumentation().getTargetContext();
        int setup = ProtectionActivity.prefs(context).getInt("setup_version", 0);
        ProtectionActivity.prefs(context).edit().putInt("setup_version", 2).commit();
        Activity activity = null;
        try (ProcessingService.NativeWork work = ProcessingService.keepWorking("Background transfer check")) {
            activity = getInstrumentation().startActivitySync(new Intent(context, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
            waitUntil(() -> ProcessingService.ready, 90_000, "Service did not start");
            waitUntil(() -> notification(context) != null
                && "Background transfer check".contentEquals(notification(context).extras.getCharSequence(Notification.EXTRA_TEXT, "")),
                15_000, "Native transfers must appear in the ongoing notification");
            Notification notification = notification(context);
            assertTrue((notification.flags & Notification.FLAG_ONGOING_EVENT) != 0);
            assertTrue(notification.extras.getBoolean(Notification.EXTRA_PROGRESS_INDETERMINATE));
            assertNotNull(notification.contentIntent);
            Activity closing = activity;
            getInstrumentation().runOnMainSync(closing::finishAndRemoveTask);
            activity = null;
            // Longer than the idle grace: closing Recents must not cancel work.
            SystemClock.sleep(70_000);
            assertEquals(0, ProcessingService.visibleScreens.get());
            assertTrue("Processing stopped after the editor was removed", ProcessingService.ready);
            assertNotNull("Background progress notification disappeared", notification(context));
        } finally {
            if (activity != null) {
                Activity closing = activity;
                getInstrumentation().runOnMainSync(closing::finishAndRemoveTask);
            }
            ProtectionActivity.prefs(context).edit().putInt("setup_version", setup).commit();
        }
        waitUntil(() -> !ProcessingService.ready, 80_000, "Idle service consumed the background time allowance");
        // Reopening after idle shutdown must restart a usable server, not reuse
        // the stale WebView session or get stuck waiting for the previous stop.
        ProtectionActivity.prefs(context).edit().putInt("setup_version", 2).commit();
        Activity reopened = getInstrumentation().startActivitySync(new Intent(context, MainActivity.class)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
        try {
            waitUntil(() -> ProcessingService.ready, 90_000, "Service did not restart after idle shutdown");
            var connection = ProcessingService.connection(ProcessingService.baseUrl + "/api/work");
            try { assertEquals(200, connection.getResponseCode()); }
            finally { connection.disconnect(); }
        } finally {
            getInstrumentation().runOnMainSync(reopened::finishAndRemoveTask);
            context.stopService(new Intent(context, ProcessingService.class));
            ProtectionActivity.prefs(context).edit().putInt("setup_version", setup).commit();
        }
    }

    private static Notification notification(Context context) {
        for (var item : context.getSystemService(NotificationManager.class).getActiveNotifications())
            if (item.getId() == 1) return item.getNotification();
        return null;
    }

    private static void waitUntil(java.util.function.BooleanSupplier check, long timeout, String message) {
        long end = SystemClock.elapsedRealtime() + timeout;
        while (!check.getAsBoolean() && SystemClock.elapsedRealtime() < end) SystemClock.sleep(250);
        assertTrue(message, check.getAsBoolean());
    }
}
