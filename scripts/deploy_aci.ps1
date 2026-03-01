param(
    [string]$ResourceGroup = "rg-rushikesh.meharwade-ai",
    [string]$Location = "eastus",
    [string]$AcrName = "",
    [string]$ImageName = "market-analyst-agent",
    [string]$ImageTag = "latest",
    [string]$ContainerGroupName = "market-analyst-agent-aci",
    [string]$DnsNameLabel = "",
    [string]$EnvFile = ".env",
    [double]$Cpu = 1.0,
    [double]$Memory = 2.0,
    [switch]$SkipBuild,
    [switch]$SkipLocalDockerBuild
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Invoke-Az([string[]]$CommandArgs) {
    $result = & az @CommandArgs --only-show-errors
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI failed: az $($CommandArgs -join ' ')"
    }
    return $result
}

function Invoke-Local([string]$Command, [string]$ErrorMessage) {
    Write-Host "  -> $Command" -ForegroundColor DarkGray
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Parse-EnvFile([string]$Path) {
    $vars = @{}
    if (-not (Test-Path $Path)) {
        throw "Env file not found: $Path"
    }

    foreach ($rawLine in Get-Content $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }

        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { continue }

        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()

        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ($key) {
            $vars[$key] = $value
        }
    }

    return $vars
}

Write-Step "Checking Azure login context"
$null = Invoke-Az @("account", "show", "--output", "none")

Write-Step "Ensuring resource group exists"
Invoke-Az @("group", "create", "--name", $ResourceGroup, "--location", $Location, "--output", "none") | Out-Null

if (-not $AcrName) {
    $AcrName = ("marketanalyst" + (Get-Date -Format "yyMMddHHmmss")).ToLower()
}

if ($AcrName -notmatch '^[a-z0-9]{5,50}$') {
    throw "AcrName must be 5-50 chars, lowercase letters/numbers only."
}

Write-Step "Ensuring ACR exists: $AcrName"
$acrExists = $true
try {
    Invoke-Az @("acr", "show", "--name", $AcrName, "--resource-group", $ResourceGroup, "--output", "none") | Out-Null
} catch {
    $acrExists = $false
}

if (-not $acrExists) {
    Invoke-Az @("acr", "create", "--resource-group", $ResourceGroup, "--name", $AcrName, "--sku", "Basic", "--admin-enabled", "true", "--location", $Location, "--output", "none") | Out-Null
} else {
    Invoke-Az @("acr", "update", "--name", $AcrName, "--resource-group", $ResourceGroup, "--admin-enabled", "true", "--output", "none") | Out-Null
}

$acrLoginServer = Invoke-Az @("acr", "show", "--name", $AcrName, "--resource-group", $ResourceGroup, "--query", "loginServer", "--output", "tsv")
$acrUsername = Invoke-Az @("acr", "credential", "show", "--name", $AcrName, "--resource-group", $ResourceGroup, "--query", "username", "--output", "tsv")
$acrPassword = Invoke-Az @("acr", "credential", "show", "--name", $AcrName, "--resource-group", $ResourceGroup, "--query", "passwords[0].value", "--output", "tsv")

if (-not $SkipLocalDockerBuild) {
    Write-Step "Compiling Docker image locally before deployment"
    $localTag = "$ImageName`:preflight"
    Invoke-Local "docker build -t $localTag ." "Local Docker build failed. Aborting deployment."
}

if (-not $SkipBuild) {
    Write-Step "Building image in ACR: $acrLoginServer/$($ImageName):$ImageTag"
    $built = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-Az @("acr", "build", "--registry", $AcrName, "--resource-group", $ResourceGroup, "--image", "$ImageName`:$ImageTag", "--no-logs", ".") | Out-Null
            $built = $true
            break
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds (5 * $attempt)
        }
    }
    if (-not $built) {
        throw "ACR build did not complete successfully."
    }
}

if (-not $DnsNameLabel) {
    $dnsSuffix = -join ((48..57) + (97..122) | Get-Random -Count 4 | ForEach-Object {[char]$_})
    $DnsNameLabel = "$ContainerGroupName-$dnsSuffix".ToLower()
}

$envVars = Parse-EnvFile -Path $EnvFile
$secureEnvPairs = @()
foreach ($k in $envVars.Keys) {
    $secureEnvPairs += "$k=$($envVars[$k])"
}

Write-Step "Replacing existing container group (if present)"
$containerExists = $true
try {
    Invoke-Az @("container", "show", "--resource-group", $ResourceGroup, "--name", $ContainerGroupName, "--output", "none") | Out-Null
} catch {
    $containerExists = $false
}

if ($containerExists) {
    Invoke-Az @("container", "delete", "--resource-group", $ResourceGroup, "--name", $ContainerGroupName, "--yes", "--output", "none") | Out-Null
}

Write-Step "Creating Azure Container Instance"
$createArgs = @(
    "container", "create",
    "--resource-group", $ResourceGroup,
    "--name", $ContainerGroupName,
    "--image", "$acrLoginServer/$ImageName`:$ImageTag",
    "--registry-login-server", $acrLoginServer,
    "--registry-username", $acrUsername,
    "--registry-password", $acrPassword,
    "--dns-name-label", $DnsNameLabel,
    "--ports", "80",
    "--ip-address", "Public",
    "--os-type", "Linux",
    "--restart-policy", "Always",
    "--cpu", "$Cpu",
    "--memory", "$Memory",
    "--secure-environment-variables"
)
$createArgs += $secureEnvPairs
$createArgs += @("--output", "none")

Invoke-Az $createArgs | Out-Null

$fqdn = (Invoke-Az @("container", "show", "--resource-group", $ResourceGroup, "--name", $ContainerGroupName, "--query", "ipAddress.fqdn", "--output", "tsv") | Out-String).Trim()
$ip = (Invoke-Az @("container", "show", "--resource-group", $ResourceGroup, "--name", $ContainerGroupName, "--query", "ipAddress.ip", "--output", "tsv") | Out-String).Trim()

Write-Step "Deployment complete"
Write-Host "Resource Group : $ResourceGroup"
Write-Host "ACR Name       : $AcrName"
Write-Host "Image          : $acrLoginServer/$ImageName`:$ImageTag"
Write-Host "Container      : $ContainerGroupName"
Write-Host "FQDN           : $fqdn"
Write-Host "IP             : $ip"
Write-Host "Frontend URL   : http://$fqdn/"
Write-Host "Backend Health : http://$fqdn/api/health"
