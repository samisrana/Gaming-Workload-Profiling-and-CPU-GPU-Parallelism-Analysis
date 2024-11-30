try {
    $temp = Get-CimInstance "MSAcpi_ThermalZoneTemperature" -Namespace "root/wmi" | 
        Select-Object -First 1 | 
        Select-Object -ExpandProperty "CurrentTemperature"
    if ($temp) {
        # Convert from tenths of Kelvin to Celsius
        $celsius = ($temp - 2732) / 10.0
        Write-Output $celsius
    } else {
        Write-Output "0"
    }
} catch {
    Write-Output "0"
}

