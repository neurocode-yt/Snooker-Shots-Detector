package com.snookerai.android;

import android.content.*;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import org.json.*;
import java.io.*;
import java.security.MessageDigest;
import java.util.function.Consumer;

/** CapCut's importer resolves MediaStore videos; publish only complete exports. */
final class CapCutMedia {
    static JSONObject publish(Context context, String jobId, JSONObject state, Consumer<String> progress) throws Exception {
        var resolver = context.getContentResolver();
        String folder = Environment.DIRECTORY_MOVIES + "/Snooker AI/CapCut/" + jobId + "/" + state.getString("generation").substring(0, 12) + "/";
        JSONArray uris = new JSONArray(), files = state.getJSONArray("files");
        Uri collection = MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY);
        for (int i = 0; i < files.length(); i++) {
            String name = files.getJSONObject(i).getString("filename");
            Uri privateUri = new Uri.Builder().scheme("content").authority(context.getPackageName() + ".capcut")
                .appendPath(jobId).appendPath(state.getString("generation")).appendPath(name).build();
            File source = CapCutProvider.resolveFile(context, privateUri);
            Uri saved = null;
            // The scoped folder and checksum make retries idempotent, including
            // death after publishing a clip but before returning its URI.
            try (var rows = resolver.query(collection, new String[]{MediaStore.Video.Media._ID},
                    MediaStore.MediaColumns.RELATIVE_PATH + "=? AND " + MediaStore.MediaColumns.DISPLAY_NAME + "=? AND " + MediaStore.MediaColumns.IS_PENDING + "=0",
                    new String[]{folder, name}, null)) {
                if (rows != null) while (rows.moveToNext()) {
                    Uri candidate = ContentUris.withAppendedId(collection, rows.getLong(0));
                    try (InputStream input = resolver.openInputStream(candidate); InputStream expected = new FileInputStream(source)) {
                        if (input != null && MessageDigest.isEqual(digest(input), digest(expected))) { saved = candidate; break; }
                    } catch (IOException ignored) { }
                }
            }
            if (saved == null) {
                ContentValues values = new ContentValues();
                values.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
                values.put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4");
                values.put(MediaStore.MediaColumns.RELATIVE_PATH, folder);
                values.put(MediaStore.MediaColumns.IS_PENDING, 1);
                saved = resolver.insert(collection, values);
                if (saved == null) throw new IOException("Could not save CapCut clip to Movies");
                try {
                    try (InputStream input = new FileInputStream(source);
                         var descriptor = resolver.openFileDescriptor(saved, "w")) {
                        if (descriptor == null) throw new IOException("Could not open CapCut clip");
                        try (FileOutputStream output = new FileOutputStream(descriptor.getFileDescriptor())) {
                            byte[] buffer = new byte[1024 * 1024];
                            int count;
                            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                            output.flush(); output.getFD().sync();
                        }
                    }
                    values.clear(); values.put(MediaStore.MediaColumns.IS_PENDING, 0);
                    if (resolver.update(saved, values, null, null) != 1) throw new IOException("Could not finish saving CapCut clip");
                } catch (Exception error) { resolver.delete(saved, null, null); throw error; }
            }
            uris.put(saved.toString());
            progress.accept("CapCut: " + (i + 1) + "/" + files.length() + " clips available in Movies");
        }
        return new JSONObject(state.toString()).put("media_uris", uris);
    }

    private static byte[] digest(InputStream input) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[1024 * 1024];
        int count;
        while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
        return digest.digest();
    }
}
