# PowerShell script to run Flask app with persistence
$appPath = Get-Location
$retryCount = 0
$maxRetries = 5

Write-Host "🚀 Starting Kidney Stone Detection Flask Server..."
Write-Host "📍 App path: $appPath"

while ($retryCount -lt $maxRetries) {
    Write-Host "`n📋 Attempt $($retryCount + 1)/$maxRetries to start the app..."
    
    try {
        # Run the Flask app
        python app.py
        
        # If we reach here, the app exited normally
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Host "⚠️  App exited, restarting in 3 seconds..."
            Start-Sleep -Seconds 3
        }
    }
    catch {
        Write-Host "❌ Error: $_"
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Host "⚠️  Retrying in 3 seconds..."
            Start-Sleep -Seconds 3
        }
    }
}

Write-Host "❌ Failed to start app after $maxRetries attempts"
