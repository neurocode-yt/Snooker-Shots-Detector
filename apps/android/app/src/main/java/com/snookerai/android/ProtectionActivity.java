package com.snookerai.android;

import android.Manifest;
import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.*;
import android.provider.Settings;

import android.widget.*;
import com.chaquo.python.Python;
import org.json.JSONObject;
import java.util.concurrent.*;

/** User-controlled Android power permissions and processing checkpoints. */
public class ProtectionActivity extends Activity {
    static final int SETUP_VERSION = 3;
    static final String REQUEST_PERMISSIONS = "request_background_permissions";
    private static final int BATTERY_REQUEST = 31, NOTIFICATION_REQUEST = 32, SETTINGS_REQUEST = 33;
    static SharedPreferences prefs(Context context) {
        return context.getSharedPreferences("protection", Context.MODE_PRIVATE);
    }
    static boolean notificationsEnabled(Context context) {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        NotificationChannel channel = manager.getNotificationChannel(ProcessingService.CHANNEL);
        return manager.areNotificationsEnabled() && (channel == null || channel.getImportance() != NotificationManager.IMPORTANCE_NONE);
    }
    static boolean configured(Context context) {
        return context.getSystemService(PowerManager.class).isIgnoringBatteryOptimizations(context.getPackageName())
            && !context.getSystemService(ActivityManager.class).isBackgroundRestricted()
            && notificationsEnabled(context) && prefs(context).getBoolean("auto_resume", true);
    }
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final ExecutorService reader = Executors.newSingleThreadExecutor();
    private LinearLayout content;
    private TextView permissions, backup;
    private boolean reading;
    private boolean permissionFlow, batteryAttempted, notificationsAttempted, permissionPending;

    private TextView text(String value, int size) {
        TextView view = new TextView(this); view.setText(value); view.setTextSize(size);
        view.setTextColor(0xffeff5f3); view.setPadding(0, 18, 0, 12); content.addView(view); return view;
    }
    private void button(String label, Runnable action) {
        Button button = new Button(this); button.setText(label); button.setAllCaps(false);
        content.addView(button, new LinearLayout.LayoutParams(-1, -2));
        button.setOnClickListener(view -> action.run());
    }
    private void open(Intent intent) {
        try { startActivity(intent); }
        catch (ActivityNotFoundException error) { Toast.makeText(this, "Open Snooker AI's settings in Android Settings", Toast.LENGTH_LONG).show(); }
    }

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        permissionFlow = saved == null ? getIntent().getBooleanExtra(REQUEST_PERMISSIONS, false) : saved.getBoolean("permission_flow");
        batteryAttempted = saved != null && saved.getBoolean("battery_attempted");
        notificationsAttempted = saved != null && saved.getBoolean("notifications_attempted");
        permissionPending = saved != null && saved.getBoolean("permission_pending");
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true); scroll.setBackgroundColor(0xff0d1517);
        content = new LinearLayout(this); content.setOrientation(LinearLayout.VERTICAL);
        int padding = (int)(20 * getResources().getDisplayMetrics().density);
        content.setPadding(padding, padding, padding, padding); scroll.addView(content); setContentView(scroll);
        text("Background & checkpoints", 25);
        text("Keep long matches running and protect saved progress and edits.", 16);
        text("Tap Allow in Android's background battery and notification prompts. This lets processing keep its CPU awake when your screen is off and show progress in notifications.", 14);
        permissions = text("Checking Android settings…", 16);
        button("Allow background processing", () -> {
            if (permissionPending) return;
            permissionFlow = true;
            batteryAttempted = notificationsAttempted = false;
            advancePermissions();
        });
        button("Enable processing notifications", this::askForNotifications);
        button("Open app battery / auto-start settings", () ->
            open(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + getPackageName()))));
        text("If your phone offers Unrestricted battery, Allow background activity or Auto-start, enable them for Snooker AI.", 14);
        text("You can press Home, swipe the editor out of Recents or turn off the screen while processing. The notification shows the current stage and percentage; tap it to return. Swiping the progress notification away does not cancel the work.", 14);
        Switch resume = new Switch(this); resume.setText("Automatically resume interrupted work");
        resume.setTextColor(0xffeff5f3); resume.setChecked(prefs(this).getBoolean("auto_resume", true));
        resume.setPadding(0, 24, 0, 24); content.addView(resume);
        resume.setOnCheckedChangeListener((button, checked) -> prefs(this).edit().putBoolean("auto_resume", checked).apply());
        text("Automatic checkpoints", 22);
        backup = text("Starting checkpoint protection...", 16);
        text("Detection saves its full state every 20 analysed samples or 15 seconds between frames, and after each completed stage and window. If Android closes the app, it continues from the latest saved checkpoint. Only unfinished work after that checkpoint may repeat.", 14);
        text("Progress and edits have three verified local recovery generations. Original videos are not duplicated. Keep your original videos separately. Clearing app data or uninstalling removes these local checkpoints.", 14);
        text("To keep processing, avoid Android's Active apps > Stop and Force stop buttons: they end the entire app. Android 15+ also limits background media processing to six hours; tap the saved-work notification to continue if that limit is reached. No app can override these controls. Reopening Snooker AI resumes saved work. Work you pause yourself stays paused.", 14);
        button("Return to editor", this::finish);
        handler.post(this::refresh);
    }

    private boolean permissionScreen(Intent intent, int request) {
        permissionPending = true;
        try { startActivityForResult(intent, request); return true; }
        catch (ActivityNotFoundException | SecurityException error) {
            permissionPending = false;
            Toast.makeText(this, "Use the app battery / auto-start settings button below", Toast.LENGTH_LONG).show();
            return false;
        }
    }

    private void advancePermissions() {
        if (!permissionFlow || permissionPending || !hasWindowFocus() || isFinishing() || isDestroyed()) return;
        if (!batteryAttempted && !getSystemService(PowerManager.class).isIgnoringBatteryOptimizations(getPackageName())) {
            batteryAttempted = true;
            if (permissionScreen(new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                    Uri.parse("package:" + getPackageName())), BATTERY_REQUEST)) return;
            // Some manufacturer builds omit the direct exemption dialog.
            if (permissionScreen(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS), BATTERY_REQUEST)) return;
        }
        if (!notificationsAttempted && !notificationsEnabled(this)) {
            notificationsAttempted = true;
            askForNotifications();
            if (permissionPending) return;
        }
        permissionFlow = false;
        if (getSystemService(ActivityManager.class).isBackgroundRestricted()) {
            new AlertDialog.Builder(this).setTitle("Allow unrestricted background use")
                .setMessage("Android still restricts this app. Open its battery settings, choose Unrestricted or Allow background activity, and enable Auto-start if your phone offers it.")
                .setPositiveButton("Open app settings", (dialog, which) -> open(new Intent(
                    Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + getPackageName()))))
                .setNegativeButton("Later", null).show();
        }
    }

    private void askForNotifications() {
        if (permissionPending) return;
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                && (!prefs(this).getBoolean("notification_permission_asked", false)
                    || shouldShowRequestPermissionRationale(Manifest.permission.POST_NOTIFICATIONS))) {
            prefs(this).edit().putBoolean("notification_permission_asked", true).apply();
            permissionPending = true;
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, NOTIFICATION_REQUEST);
        } else {
            // After repeated denial Android will not show the runtime dialog.
            // App settings still let the user enable the permission explicitly.
            boolean appEnabled = getSystemService(NotificationManager.class).areNotificationsEnabled();
            permissionScreen(new Intent(appEnabled ? Settings.ACTION_CHANNEL_NOTIFICATION_SETTINGS : Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                .putExtra(Settings.EXTRA_APP_PACKAGE, getPackageName())
                .putExtra(Settings.EXTRA_CHANNEL_ID, ProcessingService.CHANNEL), SETTINGS_REQUEST);
        }
    }

    @Override public void onWindowFocusChanged(boolean focused) {
        super.onWindowFocusChanged(focused);
        if (focused) handler.post(this::advancePermissions);
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == BATTERY_REQUEST || request == SETTINGS_REQUEST) {
            // Result codes are not authoritative for Settings screens. The next
            // step and status display query the actual Android permission state.
            permissionPending = false;
            handler.post(this::advancePermissions);
        }
    }

    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request == NOTIFICATION_REQUEST) {
            permissionPending = false;
            handler.post(this::advancePermissions);
        }
    }

    @Override protected void onSaveInstanceState(Bundle state) {
        super.onSaveInstanceState(state);
        state.putBoolean("permission_flow", permissionFlow);
        state.putBoolean("battery_attempted", batteryAttempted);
        state.putBoolean("notifications_attempted", notificationsAttempted);
        state.putBoolean("permission_pending", permissionPending);
    }

    private void refresh() {
        if (isDestroyed() || isFinishing()) return;
        boolean exempt = getSystemService(PowerManager.class).isIgnoringBatteryOptimizations(getPackageName());
        boolean restricted = getSystemService(ActivityManager.class).isBackgroundRestricted();
        boolean notifications = notificationsEnabled(this);
        permissions.setText("Battery optimisation: " + (exempt ? "Disabled ✓" : "Still enabled — allow background use")
            + "\nBackground restriction: " + (restricted ? "Restricted — check app settings" : "Not restricted ✓")
            + "\nNotifications: " + (notifications ? "Enabled ✓" : "Disabled"));
        if (Python.isStarted() && !reading) {
            reading = true;
            reader.execute(() -> {
                String local;
                try {
                    JSONObject state = new JSONObject(Python.getInstance().getModule("mobile_backup").callAttr("status").toString());
                    local = "Local recovery: " + state.optInt("protected_files") + " protected files";
                    if (!state.optString("error").isEmpty()) local += "\nLocal protection needs attention: " + state.getString("error");
                } catch (Exception failure) { local = "Local protection is starting…"; }
                final String value = local;
                handler.post(() -> { if (!isDestroyed()) backup.setText(value); reading = false; });
            });
        } else if (!Python.isStarted()) backup.setText("Local protection is starting...");
        handler.postDelayed(this::refresh, 2000);
    }

    @Override protected void onDestroy() { handler.removeCallbacksAndMessages(null); reader.shutdown(); super.onDestroy(); }
    @Override protected void onStart() { super.onStart(); ProcessingService.visibleScreens.incrementAndGet(); }
    @Override protected void onStop() { ProcessingService.visibleScreens.decrementAndGet(); super.onStop(); }
}
