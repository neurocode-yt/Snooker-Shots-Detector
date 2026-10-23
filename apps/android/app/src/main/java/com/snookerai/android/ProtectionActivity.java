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
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true); scroll.setBackgroundColor(0xff0d1517);
        content = new LinearLayout(this); content.setOrientation(LinearLayout.VERTICAL);
        int padding = (int)(20 * getResources().getDisplayMetrics().density);
        content.setPadding(padding, padding, padding, padding); scroll.addView(content); setContentView(scroll);
        text("Background & checkpoints", 25);
        text("Keep long matches running and protect saved progress and edits.", 16);
        permissions = text("Checking Android settings…", 16);
        button("Allow background battery use", () -> {
            PowerManager power = getSystemService(PowerManager.class);
            if (power.isIgnoringBatteryOptimizations(getPackageName())) {
                Toast.makeText(this, "Battery optimisation is already disabled for Snooker AI", Toast.LENGTH_LONG).show(); return;
            }
            open(new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:" + getPackageName())));
        });
        button("Enable processing notifications", () -> {
            if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
                requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 32);
            else open(new Intent(Settings.ACTION_CHANNEL_NOTIFICATION_SETTINGS)
                .putExtra(Settings.EXTRA_APP_PACKAGE, getPackageName())
                .putExtra(Settings.EXTRA_CHANNEL_ID, ProcessingService.CHANNEL));
        });
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
