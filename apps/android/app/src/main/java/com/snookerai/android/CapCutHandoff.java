package com.snookerai.android;

import android.app.*;
import android.content.*;
import android.content.pm.*;
import android.net.Uri;
import android.os.*;
import android.webkit.*;
import com.chaquo.python.Python;
import org.json.*;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;

/** Only the authenticated local WebView receives this narrow native bridge. */
public class CapCutHandoff {
    static final String PACKAGE = "com.lemon.lvoverseas";
    private final Activity activity;
    private final WebView web;
    private final ExecutorService worker;
    private final Handler handler = new Handler(Looper.getMainLooper());

    CapCutHandoff(Activity activity, WebView web, ExecutorService worker) {
        this.activity = activity; this.web = web; this.worker = worker;
    }

    @JavascriptInterface public boolean capcutAvailable() {
        try { activity.getPackageManager().getPackageInfo(PACKAGE, 0); return true; }
        catch (PackageManager.NameNotFoundException error) { return false; }
    }

    static Intent intent(Context context, String jobId, JSONObject state) throws Exception {
        ArrayList<Uri> clips = new ArrayList<>();
        JSONArray files = state.getJSONArray("files");
        for (int i = 0; i < files.length(); i++) {
            if (state.has("media_uris")) {
                clips.add(Uri.parse(state.getJSONArray("media_uris").getString(i)));
                continue;
            }
            Uri uri = new Uri.Builder().scheme("content").authority(context.getPackageName() + ".capcut")
                .appendPath(jobId).appendPath(state.getString("generation"))
                .appendPath(files.getJSONObject(i).getString("filename")).build();
            CapCutProvider.resolveFile(context, uri);
            clips.add(uri);
        }
        if (clips.isEmpty()) throw new IllegalArgumentException("No editable clips are ready");
        Intent send = new Intent(clips.size() == 1 ? Intent.ACTION_SEND : Intent.ACTION_SEND_MULTIPLE)
            .setType("video/mp4").setPackage(PACKAGE).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        if (clips.size() == 1) send.putExtra(Intent.EXTRA_STREAM, clips.get(0));
        else send.putParcelableArrayListExtra(Intent.EXTRA_STREAM, clips);
        ClipData data = ClipData.newUri(context.getContentResolver(), "Snooker shot clips", clips.get(0));
        for (int i = 1; i < clips.size(); i++) data.addItem(new ClipData.Item(clips.get(i)));
        send.setClipData(data);
        // CapCut also advertises cloud-upload/retouch share activities. Prefer
        // its declared local editor import instead of a generic package chooser.
        for (ResolveInfo target : context.getPackageManager().queryIntentActivities(send, PackageManager.GET_META_DATA)) {
            var info = target.activityInfo;
            if (info.metaData != null && "share_import_to_capcut".equals(info.metaData.getString("enter_launch"))) {
                send.setComponent(new ComponentName(info.packageName, info.name));
                break;
            }
        }
        return send;
    }

    private void notifyPage(String requestId, String message, boolean done, boolean failed) {
        JSONObject value = new JSONObject();
        try { value.put("requestId", requestId).put("message", message).put("done", done).put("failed", failed); }
        catch (JSONException ignored) { return; }
        handler.post(() -> {
            if (!activity.isDestroyed() && !activity.isFinishing())
                web.evaluateJavascript("window.dispatchEvent(new CustomEvent('snooker-capcut',{detail:" + value + "}))", null);
        });
    }

    @JavascriptInterface public void openCapCut(String jobId, String requestId) {
        if (!jobId.matches("[A-Za-z0-9_-]{1,160}") || !requestId.matches("[A-Za-z0-9_-]{1,80}")) return;
        if (!capcutAvailable()) {
            notifyPage(requestId, "Install CapCut on this phone, then tap Open in CapCut again", true, true);
            handler.post(() -> {
                if (activity.isDestroyed() || activity.isFinishing()) return;
                new AlertDialog.Builder(activity).setTitle("Install CapCut")
                    .setMessage("Install the official CapCut app to import the selected shots as separate editable clips.")
                    .setPositiveButton("Google Play", (dialog, which) -> {
                        try { activity.startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=" + PACKAGE))); }
                        catch (ActivityNotFoundException error) {
                            activity.startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse("https://play.google.com/store/apps/details?id=" + PACKAGE)));
                        }
                    }).setNegativeButton("Later", null).show();
            });
            return;
        }
        ProcessingService.NativeWork work = ProcessingService.keepWorking("Preparing clips for CapCut…");
        try { worker.execute(() -> {
            try (work) {
                var module = Python.getInstance().getModule("mobile_capcut");
                module.callAttr("start", jobId);
                JSONObject state;
                String previous = "";
                while (true) {
                    state = new JSONObject(module.callAttr("status", jobId).toString());
                    String message = state.getString("message");
                    if (!message.equals(previous)) { notifyPage(requestId, message, false, false); previous = message; }
                    if (!state.getString("status").equals("preparing")) break;
                    Thread.sleep(500);
                }
                if (!state.getString("status").equals("ready")) throw new IllegalStateException(state.getString("message"));
                state = CapCutMedia.publish(activity, jobId, state,
                    message -> notifyPage(requestId, message, false, false));
                Intent send = intent(activity, jobId, state);
                final int count = state.getInt("total");
                handler.post(() -> {
                    if (activity.isFinishing() || activity.isDestroyed()) return;
                    // Android forbids surprise app launches from the background.
                    // Completed files remain ready for the next explicit tap.
                    if (!activity.hasWindowFocus()) {
                        notifyPage(requestId, count + " clips saved. Tap Open in CapCut when you return", true, false);
                        return;
                    }
                    try {
                        activity.startActivity(send);
                        notifyPage(requestId, "Sent " + count + " separate clips to CapCut. Confirm the import there", true, false);
                    } catch (ActivityNotFoundException | SecurityException error) {
                        notifyPage(requestId, "CapCut could not accept the clips: " + error.getMessage(), true, true);
                    }
                });
            } catch (Exception error) { notifyPage(requestId, error.getMessage(), true, true); }
        }); } catch (java.util.concurrent.RejectedExecutionException error) {
            work.close();
            notifyPage(requestId, "Reopen the editor to prepare CapCut clips", true, true);
        }
    }
}
