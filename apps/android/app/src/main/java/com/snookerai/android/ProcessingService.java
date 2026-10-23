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
import java.util.concurrent.atomic.AtomicInteger;

/** The detector and local editor live in this service, independent of the screen. */
public class ProcessingService extends Service {
    public static volatile String baseUrl, token, failure;
    public static volatile boolean ready;
    static final String CHANNEL = "processing";
    static final AtomicInteger visibleScreens = new AtomicInteger();
    private static final ConcurrentHashMap<Object, String> nativeWork = new ConcurrentHashMap<>();
    private static final long IDLE_GRACE_MS = 60_000;
    // Serialize shutdown and startup even when Android creates a new Service
    // instance while the previous instance is still pausing its Python jobs.
    private static final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler handler = new Handler(Looper.getMainLooper());
    private PowerManager.WakeLock wakeLock;
    private boolean starting;
    private volatile boolean destroyed;
    private String lastStatus = "";
    private long idleSince;
    private volatile ProcessingStatus current = new ProcessingStatus(true, true, -1,
        "Snooker App", "Starting local video editor…", "Restoring saved checkpoints.");

    /** Protect document copies and CapCut publication as well as Python jobs. */
    static final class NativeWork implements AutoCloseable {
        NativeWork(String label) { nativeWork.put(this, label); }
        @Override public void close() { nativeWork.remove(this); }
    }

    static NativeWork keepWorking(String label) { return new NativeWork(label); }

    @Override public void onCreate() {
        super.onCreate();
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL, "Video processing", NotificationManager.IMPORTANCE_LOW));
        wakeLock = ((PowerManager)getSystemService(POWER_SERVICE)).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "SnookerAI:Processing");
        wakeLock.setReferenceCounted(false);
    }

    Notification notification(ProcessingStatus status) {
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(this, 0, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        Notification.Builder builder = new Notification.Builder(this, CHANNEL).setContentTitle(status.title)
            .setContentText(status.text).setStyle(new Notification.BigTextStyle().bigText(status.detail))
            .setSmallIcon(com.snookerai.android.R.drawable.ic_notification)
            .setCategory(Notification.CATEGORY_PROGRESS).setOnlyAlertOnce(true).setShowWhen(false)
            .setContentIntent(pending).setOngoing(true);
        if (status.busy) builder.setProgress(100, Math.max(0, status.percent), status.percent < 0);
        if (Build.VERSION.SDK_INT >= 31) builder.setForegroundServiceBehavior(Notification.FOREGROUND_SERVICE_IMMEDIATE);
        return builder.build();
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        int type = Build.VERSION.SDK_INT >= 35 ? ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING : ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC;
        startForeground(1, notification(current), type);
        getSystemService(NotificationManager.class).cancel(2);
        if (!starting) {
            starting = true;
            keepAwake();
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
                            if (destroyed) return;
                            if (ProtectionActivity.prefs(this).getBoolean("auto_resume", true)) {
                                try { Python.getInstance().getModule("mobile_runtime").callAttr("recover_interrupted"); }
                                catch (Exception error) { android.util.Log.w("SnookerAI", "Automatic recovery unavailable; saved checkpoints kept", error); }
                            }
                            handler.post(() -> {
                                if (destroyed) return;
                                ready = true;
                                poll();
                            });
                            return;
                        }
                        Thread.sleep(100);
                    }
                    throw new IOException("The local editor did not start within one minute");
                } catch (Exception error) {
                    android.util.Log.e("SnookerAI", "Startup failed", error);
                    handler.post(() -> {
                        if (destroyed) return;
                        failure = error.toString();
                        ready = false;
                        ResumeAfterBoot.show(this, "Snooker App could not start", "Open the app for details. Saved checkpoints are kept.");
                        stopSelf();
                    });
                }
            });
        }
        return START_STICKY;
    }

    private void keepAwake() {
        if (destroyed) return;
        // Cover startup and the gap between a new job and the next work poll.
        // Refresh independently of HTTP progress: a slow API response must
        // not let the CPU sleep in the middle of a long analysis window.
        wakeLock.acquire(10 * 60 * 1000L);
        handler.postDelayed(this::keepAwake, 5 * 60 * 1000L);
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
                JSONObject work = new JSONObject(text);
                String transfer = nativeWork.values().stream().findFirst().orElse(null);
                if (transfer == null && work.optInt("mobile_requests") > 0) transfer = "Importing video or saving changes…";
                ProcessingStatus status = ProcessingStatus.read(work.getJSONArray("items"), transfer);
                handler.post(() -> updateStatus(status));
            } catch (Exception error) { android.util.Log.w("SnookerAI", "Progress unavailable", error); }
            finally { if (connection != null) connection.disconnect(); }
            if (!destroyed) handler.postDelayed(this::poll, 5000);
        });
    }

    private void updateStatus(ProcessingStatus status) {
        // Lifecycle callbacks run on this same thread. A completed HTTP poll
        // must never reacquire a wake lock or repost after onDestroy.
        if (destroyed) return;
        current = status;
        ProtectionActivity.prefs(this).edit().putBoolean("work_pending", status.pending).apply();
        long now = SystemClock.elapsedRealtime();
        // Android 14+ lets users swipe even ongoing notifications away.
        // Dismissal must not stop processing or lose its progress display.
        NotificationManager manager = getSystemService(NotificationManager.class);
        boolean shown = java.util.Arrays.stream(manager.getActiveNotifications()).anyMatch(n -> n.getId() == 1);
        if (!status.key().equals(lastStatus) || !shown) {
            lastStatus = status.key();
            manager.notify(1, notification(status));
        }
        if (status.busy || visibleScreens.get() > 0 || !nativeWork.isEmpty()) idleSince = 0;
        else if (idleSince == 0) idleSince = now;
        else if (now - idleSince >= IDLE_GRACE_MS) {
            // Do not consume Android's media-processing time allowance after
            // processing finishes and the editor is closed.
            stopSelf();
        }
    }

    @Override public void onTimeout(int startId, int foregroundServiceType) {
        // Android 15 limits media processing services to six hours. Checkpoints
        // stay in app storage and Continue can finish when the user reopens it.
        ResumeAfterBoot.show(this, "Android paused background processing",
            "Time limit reached. Tap to continue from saved checkpoints.");
        stopSelf();
    }

    @Override public void onTaskRemoved(Intent rootIntent) {
        // Swiping the editor out of Recents is not a request to pause its jobs.
        if (!destroyed) getSystemService(NotificationManager.class).notify(1, notification(current));
        super.onTaskRemoved(rootIntent);
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
