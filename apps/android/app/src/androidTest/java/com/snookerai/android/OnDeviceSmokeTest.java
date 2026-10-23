package com.snookerai.android;

import android.test.InstrumentationTestCase;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import org.json.JSONObject;
import java.io.*;

/** Executes real decode, classic analysis, mixed export and vertical export. */
public class OnDeviceSmokeTest extends InstrumentationTestCase {
    public void testCapCutReadOnlyHandoff() throws Exception {
        var context = getInstrumentation().getTargetContext();
        String generation = "a".repeat(64);
        File clip = new File(context.getFilesDir(), "snooker/jobs/capcut-provider-check/export/mobile_capcut/" + generation + "/shot_0001.mp4");
        clip.getParentFile().mkdirs();
        try (OutputStream out = new FileOutputStream(clip)) { out.write(new byte[]{1, 2, 3, 4}); }
        var uri = android.net.Uri.parse("content://com.snookerai.android.capcut/capcut-provider-check/" + generation + "/shot_0001.mp4");
        try (InputStream in = context.getContentResolver().openInputStream(uri)) { assertEquals(1, in.read()); }
        assertEquals("video/mp4", context.getContentResolver().getType(uri));
        try {
            context.getContentResolver().openFileDescriptor(uri, "w");
            fail("Clip provider must reject writes");
        } catch (FileNotFoundException expected) { }
        try {
            CapCutProvider.resolveFile(context, android.net.Uri.parse("content://com.snookerai.android.capcut/../" + generation + "/shot_0001.mp4"));
            fail("Clip provider must reject traversal");
        } catch (FileNotFoundException expected) { }
        JSONObject state = new JSONObject().put("generation", generation)
            .put("files", new org.json.JSONArray().put(new JSONObject().put("filename", "shot_0001.mp4")));
        android.content.Intent send = CapCutHandoff.intent(context, "capcut-provider-check", state);
        assertEquals(android.content.Intent.ACTION_SEND, send.getAction());
        assertEquals("com.lemon.lvoverseas", send.getPackage());
        assertTrue((send.getFlags() & android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION) != 0);
        assertEquals(uri, send.getClipData().getItemAt(0).getUri());
        JSONObject published = CapCutMedia.publish(context, "capcut-provider-check", state, message -> { });
        JSONObject reused = CapCutMedia.publish(context, "capcut-provider-check", state, message -> { });
        assertEquals(published.getJSONArray("media_uris").toString(), reused.getJSONArray("media_uris").toString());
        var publicUri = android.net.Uri.parse(published.getJSONArray("media_uris").getString(0));
        assertEquals("media", publicUri.getAuthority());
        assertEquals(publicUri, CapCutHandoff.intent(context, "capcut-provider-check", published).getClipData().getItemAt(0).getUri());
        context.getContentResolver().delete(publicUri, null, null);
        state.getJSONArray("files").put(new JSONObject().put("filename", "shot_0001.mp4"));
        send = CapCutHandoff.intent(context, "capcut-provider-check", state);
        assertEquals(android.content.Intent.ACTION_SEND_MULTIPLE, send.getAction());
        assertEquals(2, send.getParcelableArrayListExtra(android.content.Intent.EXTRA_STREAM).size());
        assertEquals(2, send.getClipData().getItemCount());
        clip.delete();
    }

    public void testBroadcastParity() throws Exception {
        var arguments = ((android.test.InstrumentationTestRunner) getInstrumentation()).getArguments();
        String source = arguments.getString("parity_source");
        String expected = arguments.getString("parity_expected");
        assertNotNull("Supply parity_source for the external broadcast fixture", source);
        assertNotNull("Supply parity_expected for the desktop reference", expected);
        var context = getInstrumentation().getTargetContext();
        if (!Python.isStarted()) Python.start(new AndroidPlatform(context));
        String report = Python.getInstance().getModule("mobile_smoke").callAttr("check_broadcast",
            source, expected, context.getFilesDir().getAbsolutePath() + "/snooker-parity",
            context.getApplicationInfo().nativeLibraryDir).toString();
        android.util.Log.i("SnookerParity", report);
        Python.getInstance().getModule("mobile_runtime").callAttr("stop");
    }

    public void testOfflinePipeline() throws Exception {
        var context = getInstrumentation().getTargetContext();
        if (!Python.isStarted()) Python.start(new AndroidPlatform(context));
        String result = Python.getInstance().getModule("mobile_smoke").callAttr("run",
            context.getFilesDir().getAbsolutePath() + "/snooker-test",
            context.getApplicationInfo().nativeLibraryDir).toString();
        android.util.Log.i("SnookerSmoke", result);
        JSONObject report = new JSONObject(result);
        assertTrue(report.getBoolean("decoder"));
        assertTrue(report.getBoolean("shorts"));
        assertTrue(report.getInt("analysis_features") > 0);
        File fixtures = new File(context.getFilesDir(), "scoreboards");
        for (String name : getInstrumentation().getContext().getAssets().list("blue_26_31")) {
            File target = new File(fixtures, "blue_26_31/" + name);
            target.getParentFile().mkdirs();
            try (InputStream input = getInstrumentation().getContext().getAssets().open("blue_26_31/" + name);
                 OutputStream output = new FileOutputStream(target)) {
                byte[] buffer = new byte[8192];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
            }
        }
        int scoreboards = Python.getInstance().getModule("mobile_smoke")
            .callAttr("check_scoreboards", fixtures.getAbsolutePath()).toInt();
        assertEquals(12, scoreboards);
        android.util.Log.i("SnookerSmoke", "Verified " + scoreboards + " broadcast scoreboard readings");
        Python.getInstance().getModule("mobile_runtime").callAttr("stop");
    }
}
