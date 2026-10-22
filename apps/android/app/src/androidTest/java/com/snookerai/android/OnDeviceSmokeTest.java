package com.snookerai.android;

import android.test.InstrumentationTestCase;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import org.json.JSONObject;
import java.io.*;

/** Executes real decode, classic analysis, mixed export and vertical export. */
public class OnDeviceSmokeTest extends InstrumentationTestCase {
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
