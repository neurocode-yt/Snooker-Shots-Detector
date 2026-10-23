package com.snookerai.android;

import android.app.*;
import android.content.*;

/** Android 15 forbids starting mediaProcessing from BOOT_COMPLETED. */
public class ResumeAfterBoot extends BroadcastReceiver {
    @Override public void onReceive(Context context, Intent intent) {
        if (!Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction()) || !ProtectionActivity.prefs(context).getBoolean("work_pending", false)) return;
        show(context, "Snooker AI · saved work is ready", "Tap to resume processing from saved checkpoints");
    }

    static void show(Context context, String title, String text) {
        NotificationManager manager = context.getSystemService(NotificationManager.class);
        if (!manager.areNotificationsEnabled()) return;
        manager.createNotificationChannel(new NotificationChannel("recovery", "Saved work", NotificationManager.IMPORTANCE_DEFAULT));
        PendingIntent open = PendingIntent.getActivity(context, 0, new Intent(context, MainActivity.class), PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        manager.notify(2, new Notification.Builder(context, "recovery").setSmallIcon(com.snookerai.android.R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text).setStyle(new Notification.BigTextStyle().bigText(text))
            .setContentIntent(open).setAutoCancel(true).build());
    }
}
