package com.snookerai.android;

import android.app.*;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.*;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import org.json.JSONObject;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;

/** The detector and local editor live in this service, independent of the screen. */
public class ProcessingService extends Service {
    public static volatile String baseUrl, token, failure;
    public static volatile boolean ready;
    private static final String CHANNEL = "processing";
    // Serialize shutdown and startup even when Android creates a new Service
    // instance while the previous instance is still pausing its Python jobs.
    private static final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private PowerManager.WakeLock wakeLock;
    private boolean starting;
    private volatile boolean destroyed;
    private String lastStatus = "";
    private long lastWakeRefresh;

    @Override public void onCreate() {
        super.onCreate();
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL, "Video processing", NotificationManager.IMPORTANCE_LOW));
        wakeLock = ((PowerManager)getSystemService(POWER_SERVICE)).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "SnookerAI:Processing");
        wakeLock.setReferenceCounted(false);
    }

    private Notification notification(String text) {
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(this, 0, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        return new Notification.Builder(this, CHANNEL).setContentTitle("Snooker AI · on this phone")
            .setContentText(text).setSmallIcon(com.snookerai.android.R.drawable.ic_notification)
            .setContentIntent(pending).setOngoing(true).build();
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        int type = Build.VERSION.SDK_INT >= 35 ? ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING : ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC;
        startForeground(1, notification("Starting local video editor…"), type);
        if (!starting) {
            starting = true;
            failure = null;
            worker.execute(() -> {
                try {
                    if (!Python.isStarted()) Python.start(new AndroidPlatform(this));
                    JSONObject session = new JSONObject(Python.getInstance().getModule("mobile_runtime").callAttr("start",
                        getFilesDir().getAbsolutePath() + "/snooker", getApplicationInfo().nativeLibraryDir).toString());
                    baseUrl = session.getString("url");
                    token = session.getString("token");
                    for (int attempt = 0; attempt < 600; attempt++) {
                        if (Python.getInstance().getModule("mobile_runtime").callAttr("ready").toBoolean()) {
                            ready = true;
                            handler.post(this::poll);
                            return;
                        }
                        Thread.sleep(100);
                    }
                    throw new IOException("The local editor did not start within one minute");
                } catch (Exception error) {
                    failure = error.toString();
                    ready = false;
                    android.util.Log.e("SnookerAI", "Startup failed", error);
                    handler.post(() -> { getSystemService(NotificationManager.class).notify(1, notification("Could not start. Open the app for details.")); stopSelf(); });
                }
            });
        }
        return START_NOT_STICKY;
    }

    public static HttpURLConnection connection(String url) throws IOException {
        // File export URLs must refer to this authenticated in-process server.
        if (baseUrl == null || !url.startsWith(baseUrl + "/")) throw new IOException("Invalid local export URL");
        HttpURLConnection connection = (HttpURLConnection)new URL(url).openConnection();
        connection.setInstanceFollowRedirects(false);
        connection.setConnectTimeout(15000);
        connection.setReadTimeout(30000);
        connection.setRequestProperty("Cookie", "snooker_mobile=" + token);
        return connection;
    }

    private void poll() {
        if (!ready || destroyed) return;
        worker.execute(() -> {
            if (destroyed) return;
            String status = "Ready · Choose a video to begin";
            boolean busy = false;
            HttpURLConnection connection = null;
            try {
                connection = connection(baseUrl + "/api/work");
                String text;
                try (InputStream input = connection.getInputStream(); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                    byte[] buffer = new byte[8192];
                    int count;
                    while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                    text = output.toString(StandardCharsets.UTF_8.name());
                }
                var items = new JSONObject(text).getJSONArray("items");
                for (int i = 0; i < items.length(); i++) {
                    JSONObject item = items.getJSONObject(i);
                    String state = item.getString("status");
                    if (!state.equals("paused") && !state.equals("interrupted")) {
                        busy = true;
                        status = item.optString("message", state) + " · " + Math.round(item.optDouble("progress") * 100) + "%";
                        break;
                    }
                }
                long now = SystemClock.elapsedRealtime();
                if (busy && (!wakeLock.isHeld() || now - lastWakeRefresh > 5 * 60 * 1000L)) {
                    wakeLock.acquire(10 * 60 * 1000L);
                    lastWakeRefresh = now;
                }
                else if (!busy && wakeLock.isHeld()) wakeLock.release();
                if (!status.equals(lastStatus)) {
                    lastStatus = status;
                    getSystemService(NotificationManager.class).notify(1, notification(status));
                }
            } catch (Exception error) { android.util.Log.w("SnookerAI", "Progress unavailable", error); }
            finally { if (connection != null) connection.disconnect(); }
            if (!destroyed) handler.postDelayed(this::poll, 5000);
        });
    }

    @Override public void onTimeout(int startId, int foregroundServiceType) {
        // Android 15 limits media processing services to six hours. Checkpoints
        // stay in app storage and Continue can finish when the user reopens it.
        stopSelf();
    }

    @Override public void onDestroy() {
        destroyed = true;
        ready = false;
        handler.removeCallbacksAndMessages(null);
        if (wakeLock.isHeld()) wakeLock.release();
        if (Python.isStarted()) worker.execute(() -> {
            try { Python.getInstance().getModule("mobile_runtime").callAttr("stop"); }
            catch (Exception error) { android.util.Log.w("SnookerAI", "Stopping", error); }
        });
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent intent) { return null; }
}
