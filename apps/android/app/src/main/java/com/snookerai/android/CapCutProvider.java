package com.snookerai.android;

import android.content.*;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import java.io.*;
import java.util.List;

/** Read-only access to the exact clip URIs granted to CapCut; no library root. */
public class CapCutProvider extends ContentProvider {
    @Override public boolean onCreate() { return true; }

    static File resolveFile(Context context, Uri uri) throws FileNotFoundException {
        List<String> parts = uri.getPathSegments();
        if (!(context.getPackageName() + ".capcut").equals(uri.getAuthority()) || parts.size() != 3
                || !parts.get(0).matches("[A-Za-z0-9_-]{1,160}")
                || !parts.get(1).matches("[a-f0-9]{64}")
                || !parts.get(2).matches("shot_[0-9]{4,}\\.mp4"))
            throw new FileNotFoundException("Invalid CapCut clip");
        File root = new File(context.getFilesDir(), "snooker/jobs/" + parts.get(0) + "/export/mobile_capcut/" + parts.get(1));
        File file = new File(root, parts.get(2));
        try {
            if (!file.getCanonicalFile().getParentFile().equals(root.getCanonicalFile()) || !file.isFile())
                throw new FileNotFoundException("CapCut clip unavailable");
        } catch (IOException error) { throw new FileNotFoundException(error.getMessage()); }
        return file;
    }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if (!"r".equals(mode)) throw new FileNotFoundException("Read-only clip");
        return ParcelFileDescriptor.open(resolveFile(getContext(), uri), ParcelFileDescriptor.MODE_READ_ONLY);
    }
    @Override public String getType(Uri uri) { return "video/mp4"; }
    @Override public Cursor query(Uri uri, String[] projection, String selection, String[] args, String order) {
        try {
            File file = resolveFile(getContext(), uri);
            String[] columns = projection == null ? new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE} : projection;
            MatrixCursor cursor = new MatrixCursor(columns);
            Object[] values = new Object[columns.length];
            for (int i = 0; i < columns.length; i++) {
                if (OpenableColumns.DISPLAY_NAME.equals(columns[i])) values[i] = file.getName();
                else if (OpenableColumns.SIZE.equals(columns[i])) values[i] = file.length();
            }
            cursor.addRow(values);
            return cursor;
        } catch (FileNotFoundException error) { return null; }
    }
    @Override public Uri insert(Uri uri, ContentValues values) { throw new UnsupportedOperationException(); }
    @Override public int update(Uri uri, ContentValues values, String selection, String[] args) { throw new UnsupportedOperationException(); }
    @Override public int delete(Uri uri, String selection, String[] args) { throw new UnsupportedOperationException(); }
}
