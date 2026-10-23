param(
    [switch]$Test,
    [switch]$Install,
    [string]$Device
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
$previousSerial = $env:ANDROID_SERIAL
try {
    if ($Device) { $env:ANDROID_SERIAL = $Device }
    python apps/android/prepare.py
    if ($LASTEXITCODE -ne 0) { throw "Android preparation failed" }
    $tasks = @("assembleDebug", "assembleDebugAndroidTest")
    if ($Test) { $tasks += "connectedDebugAndroidTest" }
    & .\apps\android\gradlew.bat -p apps/android @tasks --console=plain
    if ($LASTEXITCODE -ne 0) { throw "Android build or device checks failed" }
    New-Item -ItemType Directory -Path dist -Force | Out-Null
    Copy-Item -LiteralPath apps/android/app/build/outputs/apk/debug/app-debug.apk -Destination dist/Snooker-App.apk -Force
    if ($Install) {
        adb install -r dist/Snooker-App.apk
        if ($LASTEXITCODE -ne 0) { throw "APK installation failed" }
    }
    Write-Output "APK: $(Join-Path $repoRoot 'dist/Snooker-App.apk')"
} finally { $env:ANDROID_SERIAL = $previousSerial; Pop-Location }
