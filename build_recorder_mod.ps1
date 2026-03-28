# recorder-mod をビルドして Slay the Spire 2 の mods フォルダにコピーするスクリプト

$ModName = "recorder-mod"
$ModJson = "recorder_mod.json"
$ProjectFile = "recorder-mod\recorder_mod.csproj"

# --- 1. Slay the Spire 2 のパスを特定 ---
$SteamPath = Get-ItemPropertyValue -Path "HKCU:\Software\Valve\Steam" -Name "SteamPath" -ErrorAction SilentlyContinue
if ($null -eq $SteamPath) {
    Write-Warning "レジストリから Steam のパスが見つかりませんでした。"
    $SteamPath = "C:\Program Files (x86)\Steam"
}

$Sts2Path = Join-Path $SteamPath "steamapps\common\Slay the Spire 2"

# パスが存在しない場合は、Eドライブなどの一般的な外部ライブラリパスを試作
if (-not (Test-Path $Sts2Path)) {
    $Sts2Path = "E:\SteamLibrary\steamapps\common\Slay the Spire 2"
}

if (-not (Test-Path $Sts2Path)) {
    Write-Error "Slay the Spire 2 のインストールフォルダが見つかりませんでした: $Sts2Path"
    Write-Host "スクリプト内の `$Sts2Path を直接編集してください。"
    exit 1
}

$ModsDir = Join-Path $Sts2Path "mods"
if (-not (Test-Path $ModsDir)) {
    Write-Host "mods フォルダを作成します: $ModsDir"
    New-Item -ItemType Directory -Path $ModsDir -Force | Out-Null
}

# --- 2. ビルドの実行 ---
Write-Host "=== $ModName をビルドしています... ===" -ForegroundColor Cyan
dotnet build $ProjectFile -c Release

if ($LASTEXITCODE -ne 0) {
    Write-Error "ビルドに失敗しました。"
    exit 1
}

# --- 3. ファイルのコピー ---
# .csproj の設定により自動的にコピーされる設定になっていますが、
# スクリプトからも確実に行うようにします。

# 出力パスの候補
$TargetDll = "recorder_mod.dll"
$Candidates = @(
    "recorder-mod\bin\Release\net9.0\$TargetDll",
    "recorder-mod\.godot\mono\temp\bin\Release\$TargetDll"
)

$DllPath = $null
foreach ($C in $Candidates) {
    if (Test-Path $C) {
        $DllPath = $C
        break
    }
}

if ($null -ne $DllPath) {
    Write-Host "ファイルをコピーしています: $DllPath -> $ModsDir" -ForegroundColor Green
    Copy-Item -Path $DllPath -Destination $ModsDir -Force
    
    $JsonSrc = "recorder-mod\$ModJson"
    if (Test-Path $JsonSrc) {
        Copy-Item -Path $JsonSrc -Destination $ModsDir -Force
    }
    
    Write-Host "=== 完了しました！ ===" -ForegroundColor Green
} else {
    Write-Error "ビルド済みの DLL が見つかりませんでした。"
    exit 1
}
