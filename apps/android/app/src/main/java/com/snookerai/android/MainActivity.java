package com.snookerai.android;

import android.Manifest;
import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.*;
import android.view.*;
import android.webkit.*;
import android.widget.*;
import java.io.*;
import java.net.HttpURLConnection;
import java.util.concurrent.*;

public class MainActivity extends Activity {
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final ExecutorService files = Executors.newSingleThreadExecutor();
    private WebView web;
    private LinearLayout screen;
    private TextView message;
    private ValueCallback<Uri[]> fileCallback;
    private String exportUrl;
    private View fullScreen;
    private WebChromeClient.CustomViewCallback fullScreenCallback;
    private boolean loaded;
    private static final int PICK_VIDEO = 10, SAVE_EXPORT = 11;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().setStatusBarColor(0xff101a1e);
        getWindow().setNavigationBarColor(0xff0d1517);
        screen = new LinearLayout(this);
        screen.setOrientation(LinearLayout.VERTICAL);
        screen.setBackgroundColor(0xff0d1517);
        screen.setOnApplyWindowInsetsListener((view, insets) -> {
            view.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            return insets.consumeSystemWindowInsets();
        });
        message = new TextView(this);
        message.setTextColor(0xffeff5f3);
        message.setTextSize(18);
        message.setPadding(28, 70, 28, 28);
        message.setText("Snooker AI\n\nStarting your on-device editor…");
        screen.addView(message);
        setContentView(screen);
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 20);
        startForegroundService(new Intent(this, ProcessingService.class));
        handler.post(this::awaitEditor);
    }

    private void awaitEditor() {
        if (isFinishing() || isDestroyed() || loaded) return;
        if (ProcessingService.failure != null) {
            message.setText("Could not start Snooker AI\n\n" + ProcessingService.failure + "\n\nClose and reopen the app to retry.");
            return;
        }
        if (!ProcessingService.ready) { handler.postDelayed(this::awaitEditor, 250); return; }
        loaded = true;
        CookieManager.getInstance().setAcceptCookie(true);
        CookieManager.getInstance().setCookie(ProcessingService.baseUrl,
            "snooker_mobile=" + ProcessingService.token + "; HttpOnly; SameSite=Strict; Path=/");
        CookieManager.getInstance().flush();
        web = new WebView(this);
        web.setBackgroundColor(0xff0d1517);
        WebSettings settings = web.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setSupportMultipleWindows(false);
        CookieManager.getInstance().setAcceptThirdPartyCookies(web, false);
        web.setWebViewClient(new WebViewClient() {
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                String url = request.getUrl().toString();
                if (url.startsWith(ProcessingService.baseUrl + "/")) return false;
                if (request.isForMainFrame() && (url.startsWith("https://") || url.startsWith("http://"))) {
                    try { startActivity(new Intent(Intent.ACTION_VIEW, request.getUrl())); }
                    catch (ActivityNotFoundException ignored) { }
                }
                return true;
            }
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                String url = request.getUrl().toString();
                if (!url.startsWith(ProcessingService.baseUrl + "/"))
                    return new WebResourceResponse("text/plain", "UTF-8", new ByteArrayInputStream(new byte[0]));
                return null;
            }
            @Override public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame()) Toast.makeText(MainActivity.this, "Local editor unavailable. Reopen Snooker AI to continue saved work.", Toast.LENGTH_LONG).show();
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override public void onShowCustomView(View view, CustomViewCallback callback) {
                if (fullScreen != null) { callback.onCustomViewHidden(); return; }
                fullScreen = view;
                fullScreenCallback = callback;
                web.setVisibility(View.GONE);
                screen.addView(view, new LinearLayout.LayoutParams(-1, -1));
            }
            @Override public void onHideCustomView() { closeFullScreen(); }
            @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = callback;
                Intent pick = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("video/*");
                pick.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, params.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE);
                try { startActivityForResult(pick, PICK_VIDEO); }
                catch (ActivityNotFoundException error) { fileCallback.onReceiveValue(null); fileCallback = null; }
                return true;
            }
            @Override public boolean onJsConfirm(WebView view, String url, String text, JsResult result) {
                if (text.contains("CapCut")) { result.cancel(); return true; }
                return super.onJsConfirm(view, url, text, result);
            }
        });
        web.setDownloadListener((url, userAgent, disposition, mimeType, length) -> {
            if (!url.startsWith(ProcessingService.baseUrl + "/")) return;
            if (exportUrl != null) { Toast.makeText(this, "Finish the current save first", Toast.LENGTH_SHORT).show(); return; }
            exportUrl = url;
            Intent save = new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
                .setType(mimeType == null ? "application/octet-stream" : mimeType)
                .putExtra(Intent.EXTRA_TITLE, URLUtil.guessFileName(url, disposition, mimeType));
            try { startActivityForResult(save, SAVE_EXPORT); }
            catch (ActivityNotFoundException error) { exportUrl = null; Toast.makeText(this, "No document provider available", Toast.LENGTH_LONG).show(); }
        });
        screen.removeView(message);
        screen.addView(web, new LinearLayout.LayoutParams(-1, -1));
        web.loadUrl(ProcessingService.baseUrl + "/");
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == PICK_VIDEO && fileCallback != null) {
            fileCallback.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(result, data));
            fileCallback = null;
        } else if (request == SAVE_EXPORT) {
            String url = exportUrl;
            exportUrl = null;
            if (result == RESULT_OK && data != null && data.getData() != null && url != null) {
                Uri destination = data.getData();
                Toast.makeText(this, "Saving export…", Toast.LENGTH_SHORT).show();
                files.execute(() -> copyExport(url, destination));
            }
        }
    }

    private void copyExport(String url, Uri destination) {
        HttpURLConnection connection = null;
        try {
            connection = ProcessingService.connection(url);
            if (connection.getResponseCode() != 200) throw new IOException("Export is unavailable (" + connection.getResponseCode() + ")");
            try (InputStream input = connection.getInputStream(); OutputStream output = getContentResolver().openOutputStream(destination, "wt")) {
                if (output == null) throw new IOException("Cannot write to the selected location");
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
            }
            handler.post(() -> {
                if (isFinishing() || isDestroyed()) {
                    Toast.makeText(getApplicationContext(), "Export saved", Toast.LENGTH_LONG).show();
                    return;
                }
                new AlertDialog.Builder(this).setTitle("Export saved")
                .setMessage("Your export is saved in the location you selected.")
                .setPositiveButton("Share", (dialog, which) -> {
                    String type = getContentResolver().getType(destination);
                    Intent share = new Intent(Intent.ACTION_SEND).setType(type == null ? "application/octet-stream" : type)
                        .putExtra(Intent.EXTRA_STREAM, destination).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    startActivity(Intent.createChooser(share, "Share highlights"));
                }).setNegativeButton("Done", null).show();
            });
        } catch (Exception error) {
            try { getContentResolver().delete(destination, null, null); }
            catch (Exception ignored) { }
            handler.post(() -> Toast.makeText(this, "Could not save export: " + error.getMessage(), Toast.LENGTH_LONG).show());
        } finally { if (connection != null) connection.disconnect(); }
    }

    @Override public void onBackPressed() {
        if (fullScreen != null) closeFullScreen();
        else if (web != null && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    private void closeFullScreen() {
        if (fullScreen == null) return;
        screen.removeView(fullScreen);
        fullScreen = null;
        web.setVisibility(View.VISIBLE);
        fullScreenCallback.onCustomViewHidden();
        fullScreenCallback = null;
    }

    @Override protected void onDestroy() {
        handler.removeCallbacksAndMessages(null);
        closeFullScreen();
        if (fileCallback != null) fileCallback.onReceiveValue(null);
        if (web != null) { screen.removeView(web); web.destroy(); }
        files.shutdown();
        super.onDestroy();
    }
}
